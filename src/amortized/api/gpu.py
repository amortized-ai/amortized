"""GPU utilization endpoint — infers GPU usage from active K8s pods."""

import contextlib
import logging
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from amortized.config import settings
from amortized.db import get_db
from amortized.db.repository import Repository

logger = logging.getLogger("amortized.api.gpu")

router = APIRouter(prefix="/api/v1/gpu", tags=["gpu"])


class GpuJobInfo(BaseModel):
    job_id: str
    job_name: str
    job_type: str
    status: str
    gpus_requested: int
    started_at: str | None = None


class GpuUtilizationResponse(BaseModel):
    available: bool
    total_gpus_in_use: int = 0
    jobs: list[GpuJobInfo] = []
    reason: str | None = None


@router.get("", response_model=GpuUtilizationResponse, operation_id="get_gpu_utilization")
async def get_gpu_utilization(
    db: aiosqlite.Connection = Depends(get_db),
) -> GpuUtilizationResponse:
    if settings.compute_backend != "kubernetes":
        return GpuUtilizationResponse(
            available=False,
            reason="requires_kubernetes_backend",
        )

    try:
        from kubernetes_asyncio import config
        from kubernetes_asyncio.client import ApiClient, CoreV1Api

        config.load_incluster_config()
        api_client = ApiClient()
    except Exception:
        logger.debug("K8s in-cluster config unavailable", exc_info=True)
        return GpuUtilizationResponse(
            available=False,
            reason="kubernetes_unavailable",
        )

    try:
        core = CoreV1Api(api_client)
        pods = await core.list_namespaced_pod(
            settings.compute_namespace,
            field_selector="status.phase=Running",
        )

        repo = Repository(db)
        gpu_jobs: list[GpuJobInfo] = []

        for pod in pods.items or []:
            gpu_count = _extract_gpu_count(pod)
            if gpu_count <= 0:
                continue

            k8s_job_name = _extract_job_name(pod)
            started_at = None
            if pod.status and pod.status.start_time:
                started_at = pod.status.start_time.isoformat()

            job_record = await _lookup_job(repo, k8s_job_name)

            gpu_jobs.append(
                GpuJobInfo(
                    job_id=job_record["id"] if job_record else "",
                    job_name=k8s_job_name,
                    job_type=job_record["type"] if job_record else "unknown",
                    status=job_record["status"] if job_record else "running",
                    gpus_requested=gpu_count,
                    started_at=started_at,
                )
            )

        return GpuUtilizationResponse(
            available=True,
            total_gpus_in_use=sum(j.gpus_requested for j in gpu_jobs),
            jobs=gpu_jobs,
        )
    except Exception:
        logger.warning("Failed to query K8s pods for GPU utilization", exc_info=True)
        return GpuUtilizationResponse(
            available=False,
            reason="kubernetes_unavailable",
        )
    finally:
        await api_client.close()


def _extract_gpu_count(pod: Any) -> int:
    total = 0
    if not pod.spec or not pod.spec.containers:
        return 0
    for container in pod.spec.containers:
        requests = (container.resources or {}) and container.resources.requests
        if requests and "nvidia.com/gpu" in requests:
            with contextlib.suppress(ValueError, TypeError):
                total += int(requests["nvidia.com/gpu"])
        limits = (container.resources or {}) and container.resources.limits
        if not requests and limits and "nvidia.com/gpu" in limits:
            with contextlib.suppress(ValueError, TypeError):
                total += int(limits["nvidia.com/gpu"])
    return total


def _extract_job_name(pod: Any) -> str:
    if pod.metadata and pod.metadata.labels:
        job_name = pod.metadata.labels.get("job-name", "")
        if job_name:
            return job_name
    if pod.metadata and pod.metadata.owner_references:
        for ref in pod.metadata.owner_references:
            if ref.kind == "Job":
                return ref.name
    return pod.metadata.name if pod.metadata else ""


async def _lookup_job(repo: Repository, k8s_job_name: str) -> dict[str, Any] | None:
    if not k8s_job_name:
        return None
    cursor = await repo.conn.execute(
        "SELECT * FROM jobs WHERE k8s_job_name = ? LIMIT 1",
        (k8s_job_name,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    import json

    d = dict(row)
    d["config"] = json.loads(d["config"]) if isinstance(d["config"], str) else d["config"]
    return d
