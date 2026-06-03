"""SSH compute backend — runs jobs on remote machines via SSH."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from amortized.backends import (
    BackendHandle,
    BackendStatus,
    Capability,
    JobSpec,
)

logger = logging.getLogger("amortized.backends.ssh")


def _require_asyncssh():  # type: ignore[no-untyped-def]
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

    async def _connect(self):  # type: ignore[no-untyped-def]
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

        conn = await self._connect()
        try:
            await conn.run(f"mkdir -p {remote_dir}", check=True)

            config_json = json.dumps({"command": spec.command, "env": spec.env})
            await conn.run(
                f"cat > {remote_dir}/config.json << 'AMORTIZED_EOF'\n{config_json}\nAMORTIZED_EOF",
                check=True,
            )

            env_exports = " ".join(f"{k}={v}" for k, v in spec.env.items())
            cmd_str = " ".join(spec.command)
            full_cmd = (
                f"cd {remote_dir} && "
                f"{env_exports + ' ' if env_exports else ''}"
                f"nohup {cmd_str} > stdout.log 2> stderr.log & echo $!"
            )

            result = await conn.run(full_cmd, check=True)
            pid = int(result.stdout.strip())  # type: ignore[union-attr]

            logger.info(
                "Started SSH job %s on %s with pid %d", spec.job_id, self._host, pid
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
        if handle.remote_pid is None:
            return BackendStatus(running=False, error="No PID recorded")

        conn = await self._connect()
        try:
            result = await conn.run(
                f"kill -0 {handle.remote_pid} 2>/dev/null && echo alive || echo dead"
            )
            output = result.stdout.strip()  # type: ignore[union-attr]

            if output == "alive":
                return BackendStatus(running=True)

            exit_result = await conn.run(
                f"wait {handle.remote_pid} 2>/dev/null; echo $?"
            )
            exit_code_str = exit_result.stdout.strip()  # type: ignore[union-attr]
            try:
                exit_code = int(exit_code_str)
            except ValueError:
                exit_code = None

            return BackendStatus(running=False, exit_code=exit_code)
        finally:
            conn.close()

    async def cancel(self, handle: BackendHandle) -> None:
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
            output = result.stdout or ""  # type: ignore[union-attr]
            for line in output.splitlines():
                yield line
        finally:
            conn.close()
