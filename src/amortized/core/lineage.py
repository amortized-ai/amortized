"""Job lineage graph traversal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amortized.models import LineageEdge, LineageNode, LineageResponse

if TYPE_CHECKING:
    from amortized.db.repository import Repository


def _extract_meta(job: dict[str, Any]) -> dict[str, Any]:
    config = job.get("config", {})
    job_type = job.get("type", "")
    meta: dict[str, Any] = {}

    if job_type == "training":
        if v := config.get("model_name_or_path"):
            meta["model"] = v
        if v := config.get("algorithm"):
            meta["algorithm"] = v
    elif job_type == "sdg":
        if v := config.get("topic"):
            meta["topic"] = v
        if v := config.get("num_records"):
            meta["num_records"] = v
        model_configs = config.get("model_configs")
        if isinstance(model_configs, list) and model_configs:
            first = model_configs[0]
            model_name = first.get("model") or first.get("model_name", "")
            if model_name:
                meta["model"] = model_name
    elif job_type == "eval":
        if v := (config.get("model_name_or_path") or config.get("model")):
            meta["model"] = v
    elif job_type in ("upload", "document"):
        if v := (config.get("original_filename") or config.get("filename")):
            meta["filename"] = v
        if v := config.get("source"):
            meta["source"] = v

    return meta


def _job_to_node(job: dict[str, Any]) -> LineageNode:
    return LineageNode(
        id=job["id"],
        type=job["type"],
        status=job["status"],
        recipe=job.get("recipe", ""),
        mlflow_run_id=job.get("mlflow_run_id", ""),
        mlflow_experiment=job.get("mlflow_experiment", ""),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        meta=_extract_meta(job),
    )


async def get_job_lineage(repo: Repository, job_id: str) -> LineageResponse | None:
    job = await repo.get_job(job_id)
    if job is None:
        return None

    nodes_by_id: dict[str, LineageNode] = {}
    edges: list[LineageEdge] = []

    nodes_by_id[job["id"]] = _job_to_node(job)

    current = job
    while current.get("parent_job_id"):
        parent_id = current["parent_job_id"]
        if parent_id in nodes_by_id:
            break
        parent = await repo.get_job(parent_id)
        if parent is None:
            break
        nodes_by_id[parent["id"]] = _job_to_node(parent)
        edges.append(LineageEdge(source=parent["id"], target=current["id"]))
        current = parent

    root_id = current["id"]

    queue = [job_id]
    while queue:
        current_id = queue.pop(0)
        children = await repo.get_children_by_parent_id(current_id)
        for child in children:
            if child["id"] not in nodes_by_id:
                nodes_by_id[child["id"]] = _job_to_node(child)
                edges.append(LineageEdge(source=current_id, target=child["id"]))
                queue.append(child["id"])

    return LineageResponse(
        nodes=list(nodes_by_id.values()),
        edges=edges,
        root_id=root_id,
        target_id=job_id,
    )
