"""SSH compute backend — runs jobs on remote machines via SSH."""

from __future__ import annotations

import contextlib
import logging
import shlex
import uuid
from collections.abc import AsyncIterator
from typing import Any

from amortized.backends import (
    BackendHandle,
    BackendStatus,
    Capability,
    JobSpec,
)

logger = logging.getLogger("amortized.backends.ssh")


class SSHBackend:
    def __init__(
        self,
        host: str,
        user: str | None = None,
        key_path: str | None = None,
        remote_base_dir: str = "~/amortized-jobs",
        name: str = "ssh",
        container_runtime: str = "podman",
    ) -> None:
        self.name = name
        self._host = host
        self._user = user
        self._key_path = key_path
        self._remote_base_dir = remote_base_dir
        self._container_runtime = container_runtime

    def capabilities(self) -> set[Capability]:
        return {Capability.GPU, Capability.LOG_STREAM, Capability.STOP}

    async def _run(self, conn: Any, cmd: str, *, check: bool = False) -> str:
        """Run a command over SSH, stripping MOTD/banner noise from stdout."""
        marker = f"__amortized_{uuid.uuid4().hex[:12]}__"
        wrapped = f"echo {marker}; {cmd}; echo {marker}"
        result = await conn.run(wrapped, check=check)
        stdout = result.stdout or ""
        start = stdout.find(marker)
        end = stdout.rfind(marker)
        if start != -1 and end != -1 and start != end:
            return stdout[start + len(marker) : end].strip()
        return stdout.strip()

    async def _connect(self) -> Any:
        import asyncssh

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

        conn = await self._connect()
        try:
            # Resolve events URL using the SSH connection's local address —
            # this is guaranteed reachable from the remote node.
            if settings.external_url:
                events_url = f"{settings.external_url.rstrip('/')}/api/v1/events/ingest"
            else:
                import socket

                local_ip = None
                try:
                    sockname = conn.get_extra_info("sockname")
                    if isinstance(sockname, tuple) and sockname[0] and sockname[0] != "127.0.0.1":
                        local_ip = sockname[0]
                except Exception:
                    pass
                host = local_ip or socket.gethostname()
                events_url = f"http://{host}:{settings.port}/api/v1/events/ingest"

            amortized_env = {
                "AMORTIZED_JOB_ID": spec.job_id,
                "AMORTIZED_WORK_DIR": remote_dir,
                "AMORTIZED_CONFIG_PATH": config_path,
                "AMORTIZED_EVENTS_URL": events_url,
            }
            if spec.resources.nodes > 1:
                amortized_env["WORLD_SIZE"] = str(spec.resources.nodes)
                amortized_env["RANK"] = "0"
                amortized_env["LOCAL_RANK"] = "0"
            merged_env = {**amortized_env, **spec.env}
            await conn.run(f"mkdir -p {remote_dir}", check=True)

            for filename, content in spec.config_files.items():
                await conn.run(
                    f"cat > {remote_dir}/{filename} << 'AMORTIZED_EOF'\n{content}\nAMORTIZED_EOF",
                    check=True,
                )

            if spec.image:
                docker_overrides = {"AMORTIZED_WORK_DIR", "AMORTIZED_CONFIG_PATH"}
                infra_flags = " ".join(
                    f"-e {k}={shlex.quote(v)}"
                    for k, v in amortized_env.items()
                    if k not in docker_overrides
                )

                secret_names: list[tuple[str, str]] = []
                for k, v in spec.env.items():
                    secret_name = f"amortized-{spec.job_id[:8]}-{k.lower().replace('_', '-')}"
                    await conn.run(
                        f"printf '%s' {shlex.quote(v)} | "
                        f"{self._container_runtime} secret create {secret_name} -",
                        check=True,
                    )
                    secret_names.append((secret_name, k))

                secret_flags = " ".join(
                    f"--secret {sname},type=env,target={target}" for sname, target in secret_names
                )

                port_flags = " ".join(
                    f"-p {host_port}:{container_port}"
                    for host_port, container_port in spec.ports.items()
                )

                home_dir = "~"
                with contextlib.suppress(Exception):
                    home_dir = await self._run(conn, "echo $HOME", check=True)
                cmd_override = ""
                if spec.command:
                    cmd_override = " " + " ".join(shlex.quote(c) for c in spec.command)
                config_mounts = " ".join(
                    f"-v {remote_dir}/{fn}:/amortized/{fn}:ro" for fn in spec.config_files
                )
                if config_mounts:
                    config_mounts += " "

                network_flag = "--network host " if not port_flags else ""
                full_cmd = (
                    f"{self._container_runtime} run -d --gpus all "
                    f"--ipc=host --pids-limit=-1 "
                    f"{network_flag}"
                    f"{port_flags + ' ' if port_flags else ''}"
                    f"-v {remote_dir}:/amortized/work "
                    f"{config_mounts}"
                    f"-v {home_dir}:{home_dir}:ro "
                    f"{infra_flags} "
                    f"{secret_flags} "
                    f"-e AMORTIZED_WORK_DIR=/amortized/work "
                    f"-e AMORTIZED_CONFIG_PATH=/amortized/config.json "
                    f"{spec.image}{cmd_override}"
                )
                container_id = await self._run(conn, full_cmd, check=True)
                logger.info(
                    "Started container job %s on %s (container %s, %d secrets)",
                    spec.job_id,
                    self._host,
                    container_id[:12],
                    len(secret_names),
                )
                return BackendHandle(
                    backend_name=self.name,
                    job_id=spec.job_id,
                    remote_pid=None,
                    remote_dir=remote_dir,
                    container_id=container_id,
                    secret_names=secret_names or None,
                )
            else:
                env_exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in merged_env.items())
                cmd_str = " ".join(spec.command)
                full_cmd = (
                    f"cd {remote_dir} && "
                    f"{env_exports + ' ' if env_exports else ''}"
                    f"nohup {cmd_str} > stdout.log 2> stderr.log & echo $!"
                )

                pid = int(await self._run(conn, full_cmd, check=True))

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
            output = await self._run(
                conn, f"kill -0 {handle.remote_pid} 2>/dev/null && echo alive || echo dead"
            )

            if output == "alive":
                return BackendStatus(running=True)

            exit_code_str = await self._run(conn, f"wait {handle.remote_pid} 2>/dev/null; echo $?")
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
            fmt = "'{{.State.Running}} {{.State.ExitCode}}'"
            output = await self._run(
                conn,
                f"{self._container_runtime} inspect --format {fmt} "
                f"{handle.container_id} 2>/dev/null || echo 'error 1'",
            )
            parts = output.split()
            if len(parts) >= 2 and parts[0] == "true":
                return BackendStatus(running=True)
            exit_code = int(parts[1]) if len(parts) >= 2 else None
            return BackendStatus(running=False, exit_code=exit_code)
        finally:
            conn.close()

    async def cleanup_secrets(self, handle: BackendHandle) -> None:
        """Remove podman secrets created for this job."""
        if not handle.secret_names:
            return
        conn = await self._connect()
        try:
            for sname, _ in handle.secret_names:
                await conn.run(f"{self._container_runtime} secret rm {sname} 2>/dev/null || true")
            logger.info("Cleaned up %d secrets for job %s", len(handle.secret_names), handle.job_id)
        finally:
            conn.close()

    async def cancel(self, handle: BackendHandle) -> None:
        if handle.container_id:
            conn = await self._connect()
            try:
                await conn.run(
                    f"{self._container_runtime} stop {handle.container_id} 2>/dev/null || true"
                )
                logger.info(
                    "Stopped container %s for job %s",
                    handle.container_id[:12],
                    handle.job_id,
                )
            finally:
                conn.close()
            await self.cleanup_secrets(handle)
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
            for filename in ("stdout.log", "stderr.log"):
                output = await self._run(
                    conn,
                    f"test -s {handle.remote_dir}/{filename}"
                    f" && cat {handle.remote_dir}/{filename}"
                    f" || true",
                )
                if not output.strip():
                    continue
                if filename == "stderr.log":
                    yield "--- stderr ---"
                for line in output.splitlines():
                    yield line
        finally:
            conn.close()
