"""GPU utilization API — live metrics via K8s pod exec into nvidia device plugin."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("amortized.api.gpu")

router = APIRouter(prefix="/api/v1/gpu", tags=["gpu"])


class GpuDevice(BaseModel):
    index: int
    name: str
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    temperature_celsius: float | None


class GpuUtilizationResponse(BaseModel):
    available: bool
    device_count: int
    devices: list[GpuDevice]
    node_name: str | None = None
    error: str | None = None


_DEVICE_PLUGIN_LABELS = [
    "app=nvidia-device-plugin-daemonset",
    "k8s-app=nvidia-device-plugin-daemonset",
    "name=nvidia-device-plugin-ds",
]

_NVIDIA_SMI_CMD = [
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
]


async def _get_k8s_clients() -> tuple[Any, Any]:
    from kubernetes_asyncio import config
    from kubernetes_asyncio.client import ApiClient, CoreV1Api

    config.load_incluster_config()  # type: ignore[no-untyped-call]
    api_client = ApiClient()
    return api_client, CoreV1Api(api_client)


async def _detect_gpu_nodes(core: Any) -> tuple[bool, int, str | None]:
    """Detect GPU capacity from node resources. Returns (available, count, node_name)."""
    nodes = await core.list_node()
    for node in nodes.items:
        capacity = node.status.capacity or {}
        gpu_count_str = capacity.get("nvidia.com/gpu", "0")
        gpu_count = int(gpu_count_str)
        if gpu_count > 0:
            return True, gpu_count, node.metadata.name
    return False, 0, None


async def _find_device_plugin_pod(core: Any) -> str | None:
    """Find the nvidia device plugin pod in kube-system by trying known label selectors."""
    for label_selector in _DEVICE_PLUGIN_LABELS:
        try:
            pods = await core.list_namespaced_pod(
                namespace="kube-system",
                label_selector=label_selector,
            )
            for pod in pods.items:
                if pod.status.phase == "Running":
                    return str(pod.metadata.name)
        except Exception:
            continue
    return None


def _parse_nvidia_smi_output(output: str) -> list[GpuDevice]:
    """Parse nvidia-smi CSV output into GpuDevice objects."""
    devices: list[GpuDevice] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            temp: float | None = float(parts[5])
        except (ValueError, IndexError):
            temp = None
        try:
            devices.append(
                GpuDevice(
                    index=int(parts[0]),
                    name=parts[1],
                    utilization_percent=float(parts[2]),
                    memory_used_mib=float(parts[3]),
                    memory_total_mib=float(parts[4]),
                    temperature_celsius=temp,
                )
            )
        except (ValueError, IndexError):
            logger.warning("Failed to parse nvidia-smi line: %s", line)
    return devices


async def _exec_nvidia_smi(api_client: Any, core: Any, pod_name: str) -> str:
    """Exec nvidia-smi inside the device plugin pod via WebSocket stream."""
    from kubernetes_asyncio.stream import WsApiClient

    ws_client = WsApiClient(configuration=api_client.configuration)
    try:
        resp = await core.connect_get_namespaced_pod_exec(
            name=pod_name,
            namespace="kube-system",
            command=_NVIDIA_SMI_CMD,
            stderr=True,
            stdout=True,
            stdin=False,
            tty=False,
            _request_timeout=10,
        )
        return str(resp)
    finally:
        await ws_client.close()


@router.get(
    "/utilization",
    response_model=GpuUtilizationResponse,
    operation_id="get_gpu_utilization",
)
async def get_gpu_utilization() -> GpuUtilizationResponse:
    try:
        api_client, core = await _get_k8s_clients()
    except Exception:
        logger.exception("Cannot connect to K8s API")
        raise HTTPException(status_code=503, detail="Cannot reach cluster API") from None

    try:
        available, device_count, node_name = await _detect_gpu_nodes(core)
    except Exception:
        logger.exception("Failed to list nodes for GPU detection")
        raise HTTPException(status_code=503, detail="Cannot reach cluster API") from None

    if not available:
        return GpuUtilizationResponse(
            available=False,
            device_count=0,
            devices=[],
            node_name=None,
        )

    pod_name = await _find_device_plugin_pod(core)
    if pod_name is None:
        return GpuUtilizationResponse(
            available=True,
            device_count=device_count,
            devices=[],
            node_name=node_name,
            error="GPU device plugin not running",
        )

    try:
        output = await _exec_nvidia_smi(api_client, core, pod_name)
        devices = _parse_nvidia_smi_output(output)
    except Exception:
        logger.exception("nvidia-smi exec failed in pod %s", pod_name)
        return GpuUtilizationResponse(
            available=True,
            device_count=device_count,
            devices=[],
            node_name=node_name,
            error="Cannot read GPU metrics",
        )

    return GpuUtilizationResponse(
        available=True,
        device_count=device_count if device_count > 0 else len(devices),
        devices=devices,
        node_name=node_name,
    )
