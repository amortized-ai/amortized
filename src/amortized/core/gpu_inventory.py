"""GPU inventory — per-GPU state from the gpu-inventory DaemonSet.

A DaemonSet on each GPU node (deployed from amortized-deploy) runs
nvidia-smi every 30s and writes ``{index, uuid, memory_free_mb,
memory_total_mb}`` per GPU into the shared ``gpu-inventory`` ConfigMap in
the ``amortized`` namespace (one key per node).

Two consumers:
- the gpu-availability API: shows each GPU's live free memory (the Deploy
  dialog lists the user's assigned GPUs with what's left on them);
- serve job building: serve pods do NOT request ``nvidia.com/gpu`` — they
  pin to an assigned GPU via ``NVIDIA_VISIBLE_DEVICES=<uuid>`` so several
  of a user's deployments can share one GPU within their quota.

Occupancy model:
- Serve pods pin explicitly, so their UUID is in the pod spec — fully
  attributable ("mine", "held by <user>").
- Training pods (and anything else) request nvidia.com/gpu and the
  device plugin assigns a UUID at runtime that does NOT show in the pod
  spec. Those GPUs are detected by memory usage instead: a GPU whose free
  memory is well below total is busy and never assigned to a new serve
  pod. Attribution is unknown, shown as "in use".
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("amortized.core.gpu_inventory")

# Shared namespace holding the inventory ConfigMap (matches the DaemonSet
# manifest in amortized-deploy).
INVENTORY_NAMESPACE = "amortized"
INVENTORY_CONFIGMAP = "gpu-inventory"

# A GPU whose free memory is this much below its total is considered busy
# (running processes hold most of the card). ~8 GB covers driver overhead
# noise while still flagging any real workload.
BUSY_THRESHOLD_MB = 8192


async def _core_v1() -> tuple[Any, Any]:
    """An (ApiClient, CoreV1Api) pair configured in-cluster; caller closes."""
    from kubernetes_asyncio import config as k8s_config
    from kubernetes_asyncio.client import ApiClient, CoreV1Api

    k8s_config.load_incluster_config()  # type: ignore[no-untyped-call]
    api_client = ApiClient()
    return api_client, CoreV1Api(api_client)


async def read_inventory() -> list[dict[str, Any]]:
    """All GPUs across nodes: [{node, index, uuid, memory_free_mb, memory_total_mb, updated}]."""
    api_client, core = await _core_v1()
    try:
        cm = await core.read_namespaced_config_map(INVENTORY_CONFIGMAP, INVENTORY_NAMESPACE)
    finally:
        await api_client.close()

    gpus: list[dict[str, Any]] = []
    for key, raw in (cm.data or {}).items():
        if not key.endswith(".json"):
            continue
        try:
            report = json.loads(raw)
        except ValueError:
            continue
        node = str(report.get("node", key[: -len(".json")]))
        for gpu in report.get("gpus", []):
            gpus.append(
                {
                    "node": node,
                    "index": int(gpu["index"]),
                    "uuid": str(gpu["uuid"]),
                    "memory_free_mb": int(gpu["memory_free_mb"]),
                    "memory_total_mb": int(gpu["memory_total_mb"]),
                    "updated": str(report.get("updated", "")),
                }
            )
    gpus.sort(key=lambda g: (g["node"], g["index"]))
    return gpus


def _pod_gpu_uuids(pod: Any) -> set[str]:
    """UUIDs a pod pins GPUs to via NVIDIA_VISIBLE_DEVICES in its spec.

    Serve pods pin explicitly — those show up here. Device-plugin
    allocations (training) do NOT (kubelet injects them at runtime), which
    is why memory usage is the fallback occupancy signal. Values that are
    not UUIDs ("all", "none", index lists like "0,1,2") are ignored: a pod
    that sees all GPUs owns none of them specifically.
    """
    # Preferred: the backend mirrors the pin as a pod annotation
    # (Secret-based env vars are not readable in the pod spec's value field).
    annotations = getattr(getattr(pod, "metadata", None), "annotations", None) or {}
    pinned = annotations.get("amortized.io/gpu-uuids")
    if pinned:
        return {p.strip() for p in str(pinned).split(",") if p.strip().startswith("GPU-")}

    uuids: set[str] = set()
    containers = getattr(getattr(pod, "spec", None), "containers", None) or []
    for container in containers:
        for env in getattr(container, "env", None) or []:
            if env.name != "NVIDIA_VISIBLE_DEVICES":
                continue
            for part in str(env.value or "").split(","):
                part = part.strip()
                if part.startswith("GPU-"):
                    uuids.add(part)
    return uuids


async def _pinned_gpu_pods() -> list[Any]:
    """Running/pending pods across all namespaces that pin specific GPUs."""
    api_client, core = await _core_v1()
    try:
        pods = (await core.list_pod_for_all_namespaces()).items or []
    finally:
        await api_client.close()
    out = []
    for pod in pods:
        phase = str(getattr(pod.status, "phase", "") or "")
        if phase not in ("Running", "Pending"):
            continue
        if _pod_gpu_uuids(pod):
            out.append(pod)
    return out


def _is_busy(gpu: dict[str, Any]) -> bool:
    return gpu["memory_total_mb"] - gpu["memory_free_mb"] > BUSY_THRESHOLD_MB


def _user_of_namespace(namespace: str) -> str:
    """Job namespace amortized-<user>-jobs -> <user>; anything else as-is."""
    if namespace.startswith("amortized-") and namespace.endswith("-jobs"):
        return namespace[len("amortized-") : -len("-jobs")]
    return namespace or "unknown"


async def _pinned_occupancy() -> dict[str, list[dict[str, Any]]]:
    """Map of GPU uuid -> pinning pods: [{pod, namespace}] (serve pods)."""
    occupancy: dict[str, list[dict[str, Any]]] = {}
    for pod in await _pinned_gpu_pods():
        holder = {
            "pod": pod.metadata.name or "",
            "namespace": getattr(pod.metadata, "namespace", "") or "",
        }
        for uuid in _pod_gpu_uuids(pod):
            occupancy.setdefault(uuid, []).append(holder)
    return occupancy


async def assign_serve_gpu(my_namespace: str, needed_gpus: int = 1) -> list[str]:
    """Pick the GPU(s) to pin a new serve pod to.

    Policy: reuse the GPUs this user's running serve pods already pin
    (deployments share a GPU within the budget), most-free first; otherwise
    take GPUs that are neither busy (memory in use — a training job, a
    host process, or someone's deployment) nor pinned by another user.
    Returns uuids.
    """
    from amortized.jobs.base import JobBuildError

    if needed_gpus < 1:
        raise JobBuildError("serve jobs need at least one GPU")

    inventory = await read_inventory()
    pinned = await _pinned_occupancy()

    mine: set[str] = set()
    for uuid, holders in pinned.items():
        if any(h["namespace"] == my_namespace for h in holders):
            mine.add(uuid)
    known = {g["uuid"] for g in inventory}
    mine &= known

    others = {
        uuid
        for uuid, holders in pinned.items()
        if any(h["namespace"] != my_namespace for h in holders)
    }

    # Unheld = not busy (nothing running on it) and not pinned by anyone.
    # GPUs my pods pin may still be busy — they are reused regardless
    # (sharing is the point; the in-pod pre-flight checks the real memory).
    unheld = [
        g for g in inventory
        if not _is_busy(g) and g["uuid"] not in pinned and g["uuid"] not in others
    ]
    by_free = {g["uuid"]: g["memory_free_mb"] for g in inventory}

    chosen = sorted(mine, key=lambda u: -by_free.get(u, 0))[:needed_gpus]
    for gpu in sorted(unheld, key=lambda g: -g["memory_free_mb"]):
        if len(chosen) >= needed_gpus:
            break
        chosen.append(gpu["uuid"])

    if len(chosen) < needed_gpus:
        raise JobBuildError(
            "not enough GPUs for this deployment — every GPU already has"
            f" running workloads (need {needed_gpus}, {len(unheld)} free)"
        )
    return chosen


async def describe_gpus(my_namespace: str) -> dict[str, Any]:
    """GPUs with live free memory + who holds them + which are mine.

    Used by the gpu-availability API: the Deploy dialog shows the user's
    assigned GPUs (or the ones a new deployment would get) with each GPU's
    actual free memory.
    """
    inventory = await read_inventory()
    pinned = await _pinned_occupancy()

    gpus = []
    for g in inventory:
        holders = pinned.get(g["uuid"], [])
        busy = _is_busy(g)
        mine = any(h["namespace"] == my_namespace for h in holders)
        if mine:
            held_by = [my_namespace and _user_of_namespace(my_namespace)]
        elif holders:
            held_by = sorted({_user_of_namespace(h["namespace"]) for h in holders})
        elif busy:
            held_by = ["in use"]
        else:
            held_by = []
        gpus.append(
            {
                "node": g["node"],
                "index": g["index"],
                "uuid": g["uuid"],
                "memory_free_mb": g["memory_free_mb"],
                "memory_total_mb": g["memory_total_mb"],
                "mine": mine,
                "busy": busy,
                "held_by": held_by,
            }
        )
    my_uuids = sorted({g["uuid"] for g in gpus if g["mine"]})
    return {
        "gpus": gpus,
        "my_uuids": my_uuids,
        "updated": max((g["updated"] for g in inventory), default=""),
    }
