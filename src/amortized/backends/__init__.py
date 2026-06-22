"""Compute backend protocol and data types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Capability(StrEnum):
    GPU = "gpu"
    MULTI_NODE = "multi_node"
    LOG_STREAM = "log_stream"
    STOP = "stop"
    RESUME = "resume"


@dataclass
class BackendHandle:
    backend_name: str
    job_id: str
    remote_pid: int | None = None
    remote_dir: str | None = None
    container_id: str | None = None
    scheduler_id: str | None = None
    secret_names: list[tuple[str, str]] | None = None


@dataclass
class BackendStatus:
    running: bool
    exit_code: int | None = None
    error: str | None = None


@dataclass
class Resources:
    gpus: int = 1
    gpu_type: str | None = None
    cpus: int | None = None
    memory_gb: int | None = None
    nodes: int = 1


@dataclass
class S3Download:
    s3_uri: str
    local_path: str
    is_directory: bool = False


@dataclass
class JobSpec:
    job_id: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    work_dir: str = "."
    image: str | None = None
    timeout: int | None = None
    resources: Resources = field(default_factory=Resources)
    ports: dict[int, int] = field(default_factory=dict)
    config_files: dict[str, str] = field(default_factory=dict)
    s3_downloads: list[S3Download] = field(default_factory=list)


@runtime_checkable
class ComputeBackend(Protocol):
    name: str

    def capabilities(self) -> set[Capability]: ...
    async def submit(self, spec: JobSpec) -> BackendHandle: ...
    async def status(self, handle: BackendHandle) -> BackendStatus: ...
    async def cancel(self, handle: BackendHandle) -> None: ...
    def logs(self, handle: BackendHandle) -> AsyncIterator[str]: ...
