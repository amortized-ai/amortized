"""GPU utilization endpoint."""

import csv
import io
import logging
import shutil
import subprocess

from fastapi import APIRouter

from amortized.models import GpuNodeMetrics, GpuUtilizationResponse

logger = logging.getLogger("amortized.api.gpu")

router = APIRouter(prefix="/api/v1/gpu", tags=["gpu"])


def _query_nvidia_smi() -> list[GpuNodeMetrics] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed: %s", result.stderr.strip())
            return None
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("nvidia-smi error: %s", exc)
        return None

    nodes: list[GpuNodeMetrics] = []
    reader = csv.reader(io.StringIO(result.stdout.strip()))
    for row in reader:
        if len(row) < 6:
            continue
        nodes.append(
            GpuNodeMetrics(
                index=int(row[0].strip()),
                name=row[1].strip(),
                utilization_pct=float(row[2].strip()),
                memory_used_mb=float(row[3].strip()),
                memory_total_mb=float(row[4].strip()),
                temperature_c=float(row[5].strip()),
            )
        )
    return nodes


def _query_torch() -> list[GpuNodeMetrics] | None:
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    nodes: list[GpuNodeMetrics] = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        utilization = float(torch.cuda.utilization(i))
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        nodes.append(
            GpuNodeMetrics(
                index=i,
                name=name,
                utilization_pct=utilization,
                memory_used_mb=used / (1024 * 1024),
                memory_total_mb=total / (1024 * 1024),
                temperature_c=0.0,
            )
        )
    return nodes


@router.get(
    "/utilization",
    response_model=GpuUtilizationResponse,
    operation_id="get_gpu_utilization",
)
async def get_gpu_utilization() -> GpuUtilizationResponse:
    nodes = _query_nvidia_smi()
    if nodes is None:
        nodes = _query_torch()
    if nodes is None:
        nodes = []
    return GpuUtilizationResponse(nodes=nodes)
