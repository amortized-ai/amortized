"""SSH compute backend — runs jobs on remote machines via SSH."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from amortized.backends import (
    BackendHandle,
    BackendStatus,
    Capability,
    JobSpec,
)

logger = logging.getLogger("amortized.backends.ssh")


def _require_asyncssh() -> Any:
    try:
        import asyncssh

        return asyncssh
    except ImportError:
        raise ImportError(
            "asyncssh is required for SSHBackend. Install with: pip install amortized[ssh]"
        ) from None


class SSHBackend:
    name = "ssh"

    def __init__(
        self,
        host: str,
        user: str | None = None,
        key_path: str | None = None,
        remote_base_dir: str = "~/amortized-jobs",
    ) -> None:
        self._host = host
        self._user = user
        self._key_path = key_path
        self._remote_base_dir = remote_base_dir

    def capabilities(self) -> set[Capability]:
        return {Capability.GPU, Capability.LOG_STREAM, Capability.STOP}

    async def _connect(self) -> Any:
        asyncssh = _require_asyncssh()
        kwargs: dict[str, object] = {
            "host": self._host,
            "known_hosts": None,
        }
        if self._user:
            kwargs["username"] = self._user
        if self._key_path:
            kwargs["client_keys"] = [self._key_path]
        return await asyncssh.connect(**kwargs)

    async def submit(self, spec: JobSpec) -> BackendHandle:
        remote_dir = f"{self._remote_base_dir}/{spec.job_id}"
        config_path = f"{remote_dir}/config.json"

        from amortized.config import settings

        control_plane_host = settings.host if settings.host != "0.0.0.0" else "localhost"
        control_plane_port = settings.port

        amortized_env = {
            "AMORTIZED_JOB_ID": spec.job_id,
            "AMORTIZED_WORK_DIR": remote_dir,
            "AMORTIZED_CONFIG_PATH": config_path,
            "AMORTIZED_EVENTS_URL": (
                f"http://{control_plane_host}:{control_plane_port}/api/v1/events/ingest"
            ),
        }
        if spec.resources.nodes > 1:
            amortized_env["WORLD_SIZE"] = str(spec.resources.nodes)
            amortized_env["RANK"] = "0"
            amortized_env["LOCAL_RANK"] = "0"
        filtered_spec_env = {k: v for k, v in spec.env.items() if k != "_config"}
        merged_env = {**amortized_env, **filtered_spec_env}

        conn = await self._connect()
        try:
            await conn.run(f"mkdir -p {remote_dir}", check=True)

            config_data: dict[str, object] = {
                "config": spec.env.get("_config", {}),
                "artifacts": {},
            }
            config_json = json.dumps(config_data)
            await conn.run(
                f"cat > {config_path} << 'AMORTIZED_EOF'\n{config_json}\nAMORTIZED_EOF",
                check=True,
            )

            if spec.image:
                docker_env_flags = " ".join(f"-e {k}={v}" for k, v in amortized_env.items())
                docker_env_flags += "".join(f" -e {k}={v}" for k, v in filtered_spec_env.items())
                full_cmd = (
                    f"docker run -d --gpus all "
                    f"-v {remote_dir}:/amortized/work "
                    f"-v {config_path}:/amortized/config.json "
                    f"{docker_env_flags} "
                    f"-e AMORTIZED_WORK_DIR=/amortized/work "
                    f"-e AMORTIZED_CONFIG_PATH=/amortized/config.json "
                    f"{spec.image} "
                    f"python -m runner"
                )
                result = await conn.run(full_cmd, check=True)
                container_id = result.stdout.strip()
                logger.info(
                    "Started Docker job %s on %s (container %s)",
                    spec.job_id,
                    self._host,
                    container_id[:12],
                )
                return BackendHandle(
                    backend_name=self.name,
                    job_id=spec.job_id,
                    remote_pid=None,
                    remote_dir=remote_dir,
                    container_id=container_id,
                )
            else:
                env_exports = " ".join(f"{k}={v}" for k, v in merged_env.items())
                cmd_str = " ".join(spec.command)
                full_cmd = (
                    f"cd {remote_dir} && "
                    f"{env_exports + ' ' if env_exports else ''}"
                    f"nohup {cmd_str} > stdout.log 2> stderr.log & echo $!"
                )

                result = await conn.run(full_cmd, check=True)
                pid = int(result.stdout.strip())

                logger.info(
                    "Started SSH job %s on %s with pid %d",
                    spec.job_id,
                    self._host,
                    pid,
                )

                return BackendHandle(
                    backend_name=self.name,
                    job_id=spec.job_id,
                    remote_pid=pid,
                    remote_dir=remote_dir,
                )
        finally:
            conn.close()

    async def status(self, handle: BackendHandle) -> BackendStatus:
        if handle.container_id:
            return await self._docker_status(handle)

        if handle.remote_pid is None:
            return BackendStatus(running=False, error="No PID recorded")

        conn = await self._connect()
        try:
            result = await conn.run(
                f"kill -0 {handle.remote_pid} 2>/dev/null && echo alive || echo dead"
            )
            output = result.stdout.strip()

            if output == "alive":
                return BackendStatus(running=True)

            exit_result = await conn.run(f"wait {handle.remote_pid} 2>/dev/null; echo $?")
            exit_code_str = exit_result.stdout.strip()
            try:
                exit_code = int(exit_code_str)
            except ValueError:
                exit_code = None

            return BackendStatus(running=False, exit_code=exit_code)
        finally:
            conn.close()

    async def _docker_status(self, handle: BackendHandle) -> BackendStatus:
        conn = await self._connect()
        try:
            result = await conn.run(
                f"docker inspect --format '{{{{.State.Running}}}} {{{{.State.ExitCode}}}}' "
                f"{handle.container_id} 2>/dev/null || echo 'error 1'"
            )
            parts = result.stdout.strip().split()
            if len(parts) >= 2 and parts[0] == "true":
                return BackendStatus(running=True)
            exit_code = int(parts[1]) if len(parts) >= 2 else None
            return BackendStatus(running=False, exit_code=exit_code)
        finally:
            conn.close()

    async def cancel(self, handle: BackendHandle) -> None:
        if handle.container_id:
            conn = await self._connect()
            try:
                await conn.run(f"docker stop {handle.container_id} 2>/dev/null || true")
                logger.info(
                    "Stopped Docker container %s for job %s",
                    handle.container_id[:12],
                    handle.job_id,
                )
            finally:
                conn.close()
            return

        if handle.remote_pid is None:
            return

        conn = await self._connect()
        try:
            await conn.run(f"kill {handle.remote_pid} 2>/dev/null || true")
            logger.info("Cancelled SSH job %s (pid %d)", handle.job_id, handle.remote_pid)
        finally:
            conn.close()

    async def logs(self, handle: BackendHandle) -> AsyncIterator[str]:
        if handle.remote_dir is None:
            return

        conn = await self._connect()
        try:
            result = await conn.run(f"cat {handle.remote_dir}/stdout.log 2>/dev/null || true")
            output = result.stdout or ""
            for line in output.splitlines():
                yield line
        finally:
            conn.close()
