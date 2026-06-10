"""Slurm compute backend — submits jobs to HPC clusters via sbatch over SSH."""

from __future__ import annotations

import logging
import re
import shlex
import textwrap
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from amortized.backends import (
    BackendHandle,
    BackendStatus,
    Capability,
    JobSpec,
)

logger = logging.getLogger("amortized.backends.slurm")


def _require_asyncssh() -> Any:
    try:
        import asyncssh

        return asyncssh
    except ImportError:
        raise ImportError(
            "asyncssh is required for SlurmBackend. Install with: pip install amortized[ssh]"
        ) from None


@dataclass
class SlurmProfile:
    """Pre-configured settings for a specific HPC cluster."""

    name: str
    partition: str
    account: str = ""
    module_loads: list[str] = field(default_factory=list)
    remote_base_dir: str = "$HOME/amortized-jobs"
    gpu_resource_format: str = "gres:gpu:{count}"
    scheduler: str = "slurm"


FRONTIER = SlurmProfile(
    name="frontier",
    partition="batch",
    account="",
    module_loads=["rocm", "python"],
    remote_base_dir="$MEMBERWORK/amortized-jobs",
    gpu_resource_format="gres:gpu:{count}",
)

PERLMUTTER = SlurmProfile(
    name="perlmutter",
    partition="gpu",
    account="",
    module_loads=["python", "cuda"],
    remote_base_dir="$SCRATCH/amortized-jobs",
    gpu_resource_format="gres:gpu:a100:{count}",
)

POLARIS = SlurmProfile(
    name="polaris",
    partition="prod",
    account="",
    module_loads=["conda", "cuda"],
    remote_base_dir="/eagle/amortized-jobs",
    gpu_resource_format="gres:gpu:{count}",
)

PROFILES: dict[str, SlurmProfile] = {
    "frontier": FRONTIER,
    "perlmutter": PERLMUTTER,
    "polaris": POLARIS,
}


class SlurmBackend:
    name = "slurm"

    def __init__(
        self,
        host: str,
        username: str | None = None,
        profile: SlurmProfile | None = None,
    ) -> None:
        self._host = host
        self._username = username
        self._profile = profile or SlurmProfile(name="default", partition="batch")

    def capabilities(self) -> set[Capability]:
        return {Capability.GPU, Capability.LOG_STREAM, Capability.STOP}

    def _generate_sbatch_script(self, spec: JobSpec) -> str:
        profile = self._profile
        gpu_count = spec.resources.gpus
        gpu_line = ""
        if gpu_count > 0:
            resource = profile.gpu_resource_format.format(count=gpu_count)
            gpu_line = f"#SBATCH --{resource}"

        module_lines = "\n".join(f"module load {m}" for m in profile.module_loads)

        env_lines = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in spec.env.items())

        cmd = " ".join(spec.command)

        account_line = f"#SBATCH --account={profile.account}" if profile.account else ""

        script = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=amortized-{spec.job_id[:8]}
            #SBATCH --partition={profile.partition}
            {account_line}
            {gpu_line}
            #SBATCH --output={spec.work_dir}/stdout.log
            #SBATCH --error={spec.work_dir}/stderr.log
            #SBATCH --time=24:00:00

            {module_lines}
            {env_lines}

            cd {spec.work_dir}
            {cmd}
        """)
        return script

    async def submit(self, spec: JobSpec) -> BackendHandle:
        asyncssh = _require_asyncssh()

        connect_kwargs: dict[str, Any] = {"host": self._host, "known_hosts": None}
        if self._username:
            connect_kwargs["username"] = self._username

        remote_dir = f"{self._profile.remote_base_dir}/{spec.job_id}"

        async with asyncssh.connect(**connect_kwargs) as conn:
            await conn.run(f"mkdir -p {remote_dir}", check=True)

            sbatch_script = self._generate_sbatch_script(
                JobSpec(
                    job_id=spec.job_id,
                    command=spec.command,
                    env=spec.env,
                    work_dir=remote_dir,
                    image=spec.image,
                    resources=spec.resources,
                )
            )

            script_path = f"{remote_dir}/job.sbatch"
            write_cmd = f"cat > {script_path} << 'SBATCH_EOF'\n{sbatch_script}\nSBATCH_EOF"
            await conn.run(write_cmd, check=True)

            result = await conn.run(f"sbatch {script_path}", check=True)
            stdout = result.stdout.strip() if result.stdout else ""
            match = re.search(r"Submitted batch job (\d+)", stdout)
            scheduler_id = match.group(1) if match else ""

        logger.info("Submitted Slurm job %s (scheduler_id=%s)", spec.job_id, scheduler_id)

        return BackendHandle(
            backend_name=self.name,
            job_id=spec.job_id,
            remote_dir=remote_dir,
            scheduler_id=scheduler_id,
        )

    async def status(self, handle: BackendHandle) -> BackendStatus:
        if not handle.scheduler_id:
            return BackendStatus(running=False, error="No Slurm job ID")

        asyncssh = _require_asyncssh()

        connect_kwargs: dict[str, Any] = {"host": self._host, "known_hosts": None}
        if self._username:
            connect_kwargs["username"] = self._username

        try:
            async with asyncssh.connect(**connect_kwargs) as conn:
                result = await conn.run(
                    f"squeue -j {handle.scheduler_id} -h -o %T",
                    check=False,
                )
                stdout = (result.stdout or "").strip()

                if stdout in ("RUNNING", "COMPLETING"):
                    return BackendStatus(running=True)
                if stdout in ("PENDING", "CONFIGURING"):
                    return BackendStatus(running=True)

                sacct_result = await conn.run(
                    f"sacct -j {handle.scheduler_id} -n -o State -P",
                    check=False,
                )
                sacct_out = (sacct_result.stdout or "").strip().split("\n")[0]

                if sacct_out == "COMPLETED":
                    return BackendStatus(running=False, exit_code=0)
                if sacct_out == "CANCELLED":
                    return BackendStatus(running=False, exit_code=-1)

                return BackendStatus(running=False, exit_code=1, error=f"Slurm state: {sacct_out}")
        except Exception as exc:
            return BackendStatus(running=False, error=str(exc))

    async def cancel(self, handle: BackendHandle) -> None:
        if not handle.scheduler_id:
            return

        asyncssh = _require_asyncssh()

        connect_kwargs: dict[str, Any] = {"host": self._host, "known_hosts": None}
        if self._username:
            connect_kwargs["username"] = self._username

        async with asyncssh.connect(**connect_kwargs) as conn:
            await conn.run(f"scancel {handle.scheduler_id}", check=False)

        logger.info("Cancelled Slurm job %s", handle.scheduler_id)

    async def logs(self, handle: BackendHandle) -> AsyncIterator[str]:
        if not handle.remote_dir:
            return

        asyncssh = _require_asyncssh()

        connect_kwargs: dict[str, Any] = {"host": self._host, "known_hosts": None}
        if self._username:
            connect_kwargs["username"] = self._username

        try:
            async with asyncssh.connect(**connect_kwargs) as conn:
                result = await conn.run(
                    f"cat {handle.remote_dir}/stdout.log 2>/dev/null",
                    check=False,
                )
                if result.stdout:
                    for line in result.stdout.splitlines():
                        yield line
        except Exception:
            return
