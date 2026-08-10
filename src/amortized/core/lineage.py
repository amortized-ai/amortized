"""Job lineage graph traversal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amortized.models import LineageChainSummary, LineageEdge, LineageNode, LineageResponse

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


async def list_lineage_chains(
    repo: Repository,
    *,
    job_type: str = "",
    status: str = "",
) -> list[LineageChainSummary]:
    all_jobs = await repo.list_jobs()

    jobs_by_id: dict[str, dict[str, Any]] = {j["id"]: j for j in all_jobs}
    children_by_parent: dict[str, list[str]] = {}
    for j in all_jobs:
        pid = j.get("parent_job_id", "")
        if pid:
            children_by_parent.setdefault(pid, []).append(j["id"])

    roots: list[str] = []
    for j in all_jobs:
        pid = j.get("parent_job_id", "")
        if not pid or pid not in jobs_by_id:
            roots.append(j["id"])

    chains: list[LineageChainSummary] = []
    for root_id in roots:
        nodes_by_id: dict[str, LineageNode] = {}
        edges: list[LineageEdge] = []

        queue = [root_id]
        while queue:
            current_id = queue.pop(0)
            if current_id not in jobs_by_id or current_id in nodes_by_id:
                continue
            job = jobs_by_id[current_id]
            nodes_by_id[current_id] = _job_to_node(job)
            for child_id in children_by_parent.get(current_id, []):
                edges.append(LineageEdge(source=current_id, target=child_id))
                queue.append(child_id)

        if len(nodes_by_id) < 2:
            continue

        chain_jobs = [jobs_by_id[nid] for nid in nodes_by_id]

        if job_type and not any(j.get("type") == job_type for j in chain_jobs):
            continue
        if status and not any(j.get("status") == status for j in chain_jobs):
            continue

        timestamps = [j.get("created_at", "") for j in chain_jobs if j.get("created_at")]
        completed = [j.get("completed_at", "") for j in chain_jobs if j.get("completed_at")]
        earliest = min(timestamps) if timestamps else ""
        latest = max(completed) if completed else (max(timestamps) if timestamps else "")

        leaf_jobs = sorted(chain_jobs, key=lambda j: j.get("created_at", ""), reverse=True)
        latest_status = leaf_jobs[0]["status"] if leaf_jobs else ""

        root_job = jobs_by_id[root_id]
        recipe = root_job.get("recipe", "")
        name = recipe if recipe else f"{root_job.get('type', 'job')} {root_id[:8]}"

        lineage = LineageResponse(
            nodes=list(nodes_by_id.values()),
            edges=edges,
            root_id=root_id,
            target_id=root_id,
        )

        chains.append(
            LineageChainSummary(
                chain_id=root_id,
                name=name,
                job_count=len(nodes_by_id),
                latest_status=latest_status,
                created_at=earliest,
                updated_at=latest,
                lineage=lineage,
            )
        )

    chains.sort(key=lambda c: c.created_at, reverse=True)
    return chains
