"""Local compute backend — runs jobs as subprocesses on the current machine."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from amortized.backends import (
    BackendHandle,
    BackendStatus,
    Capability,
    JobSpec,
)

logger = logging.getLogger("amortized.backends.local")


class LocalBackend:
    name = "local"

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def capabilities(self) -> set[Capability]:
        return {Capability.LOG_STREAM, Capability.STOP}

    async def submit(self, spec: JobSpec) -> BackendHandle:
        work_dir = os.path.expanduser(spec.work_dir)
        os.makedirs(work_dir, exist_ok=True)

        config_path = os.path.join(work_dir, "config.json")
        config_data: dict[str, object] = {"config": spec.env.get("_config", {}), "artifacts": {}}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Write special env var payloads to files in work_dir
        if "_run_config" in spec.env:
            with open(os.path.join(work_dir, "config.yaml"), "w") as f:
                f.write(spec.env["_run_config"])
        if "_synth_config" in spec.env:
            with open(os.path.join(work_dir, "synth_config.yaml"), "w") as f:
                f.write(spec.env["_synth_config"])

        # Transform container paths to local paths in the command
        command = [
            part.replace("/amortized/work/config.yaml", os.path.join(work_dir, "config.yaml"))
            .replace(
                "/amortized/work/synth_config.yaml", os.path.join(work_dir, "synth_config.yaml")
            )
            .replace("python3.11", sys.executable)
            for part in spec.command
        ]

        stdout_path = os.path.join(work_dir, "stdout.log")
        stderr_path = os.path.join(work_dir, "stderr.log")

        stdout_file = open(stdout_path, "w")  # noqa: SIM115
        stderr_file = open(stderr_path, "w")  # noqa: SIM115

        amortized_env = {
            "AMORTIZED_JOB_ID": spec.job_id,
            "AMORTIZED_WORK_DIR": work_dir,
            "AMORTIZED_CONFIG_PATH": config_path,
        }
        filtered_spec_env = {
            k: v
            for k, v in spec.env.items()
            if k not in ("_config", "_run_config", "_synth_config")
        }
        env = {**os.environ, **amortized_env, **filtered_spec_env}

        # Prepend venv bin dir to PATH so venv tools (trl, torchrun, accelerate, etc.) are found
        venv_bin = os.path.dirname(sys.executable)
        env["PATH"] = venv_bin + ":" + env.get("PATH", os.environ.get("PATH", ""))

        proc = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            cwd=work_dir if Path(work_dir).is_dir() else None,
        )

        self._processes[spec.job_id] = proc

        logger.info("Started local job %s with pid %d", spec.job_id, proc.pid)

        return BackendHandle(
            backend_name=self.name,
            job_id=spec.job_id,
            remote_pid=proc.pid,
            remote_dir=work_dir,
        )

    async def status(self, handle: BackendHandle) -> BackendStatus:
        proc = self._processes.get(handle.job_id)
        if proc is not None:
            retcode = proc.poll()
            if retcode is None:
                return BackendStatus(running=True)
            return BackendStatus(running=False, exit_code=retcode)

        if handle.remote_pid is not None:
            try:
                os.kill(handle.remote_pid, 0)
                return BackendStatus(running=True)
            except OSError:
                return BackendStatus(running=False)

        return BackendStatus(running=False, error="Process not found")

    async def cancel(self, handle: BackendHandle) -> None:
        proc = self._processes.get(handle.job_id)
        if proc is not None:
            proc.terminate()
            loop = asyncio.get_event_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, proc.wait),
                    timeout=5.0,
                )
            except TimeoutError:
                proc.kill()
            self._processes.pop(handle.job_id, None)
            return

        if handle.remote_pid is not None:
            with contextlib.suppress(OSError):
                os.kill(handle.remote_pid, signal.SIGTERM)

    async def logs(self, handle: BackendHandle) -> AsyncIterator[str]:
        if handle.remote_dir is None:
            return

        stdout_path = os.path.join(handle.remote_dir, "stdout.log")
        if not os.path.exists(stdout_path):
            return

        with open(stdout_path) as f:
            for line in f:
                yield line.rstrip("\n")
