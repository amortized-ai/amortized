"""Unit tests for field-driven job-dependency enforcement (core/lineage.py) and the
upstream ancestry traversal."""

import asyncio
import json
from typing import Any, ClassVar

from amortized.core import lineage


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _fetch_from(jobs: dict[str, dict[str, Any]]) -> lineage.JobFetch:
    async def _fetch(job_id: str) -> dict[str, Any] | None:
        return jobs.get(job_id)

    return _fetch


class TestFieldDrivenGate:
    def test_gated_fields_are_the_upstream_job_references(self) -> None:
        # The enforcement universe is exactly the config keys that reference an upstream
        # job; MLflow-run fields (data_run_id / eval_data_run_id) are intentionally out.
        assert set(lineage.GATED_FIELDS) == {"parent_job_id", "training_job_id"}
        assert "data_run_id" not in lineage.GATED_FIELDS

    def test_present_gated_fields_picks_referenced_jobs(self) -> None:
        refs = lineage.present_gated_fields(
            {"parent_job_id": "sdg-1", "training_job_id": "", "data_run_id": "run-9", "x": 1}
        )
        # only non-empty gated job references; data_run_id is a run, not gated
        assert [(f, j) for f, j, _ in refs] == [("parent_job_id", "sdg-1")]

    def test_present_gated_fields_covers_any_job(self) -> None:
        # a job type "beyond the graph" still surfaces its referenced prerequisites
        refs = lineage.present_gated_fields({"training_job_id": "t-1"})
        assert [(f, j) for f, j, _ in refs] == [("training_job_id", "t-1")]

    def test_present_gated_fields_empty_when_no_references(self) -> None:
        # a dispatch that references no upstream job (e.g. a from-scratch SDG) gates
        # nothing — there is no special-casing by job type.
        assert lineage.present_gated_fields({"num_records": 1000}) == []


class TestAncestry:
    # sdg2 ──(eval dataset)──┐
    # sdg1 ──parent──▶ training ──(model)──▶ eval ◀── parent ── sdg2
    _JOBS: ClassVar[dict[str, dict[str, Any]]] = {
        "sdg1": {"id": "sdg1", "type": "sdg", "mlflow_run_id": "run-sdg1"},
        "sdg2": {"id": "sdg2", "type": "sdg", "mlflow_run_id": "run-sdg2"},
        "train1": {"id": "train1", "type": "training", "parent_job_id": "sdg1"},
        "eval1": {
            "id": "eval1",
            "type": "eval",
            "parent_job_id": "sdg2",
            "config": {"training_job_id": "train1"},
        },
    }

    def test_job_ref_reads_column_and_config(self) -> None:
        job = {"parent_job_id": "p", "config": {"training_job_id": "t"}}
        assert lineage.job_ref(job, "parent_job_id") == "p"
        assert lineage.job_ref(job, "training_job_id") == "t"  # from config
        assert lineage.job_ref(job, "data_run_id") == ""

    def test_job_ref_parses_json_string_config(self) -> None:
        job = {"config": json.dumps({"training_job_id": "t9"})}
        assert lineage.job_ref(job, "training_job_id") == "t9"

    def test_ancestors_walks_full_chain(self) -> None:
        fetch = _fetch_from(self._JOBS)
        got = _run(self._collect(self._JOBS["eval1"], fetch))
        # eval's references in GATED_FIELDS order: parent_job_id (sdg2), then
        # training_job_id (train1), whose own parent (sdg1) is reached transitively.
        assert [j["id"] for j in got] == ["sdg2", "train1", "sdg1"]

    def test_nearest_ancestor_single_hop(self) -> None:
        fetch = _fetch_from(self._JOBS)
        sdg = _run(lineage.nearest_ancestor(self._JOBS["train1"], "sdg", fetch))
        assert sdg is not None and sdg["id"] == "sdg1"

    def test_nearest_ancestor_prefers_closest(self) -> None:
        # eval's own dataset parent (sdg2) is closer than the model's training SDG (sdg1)
        fetch = _fetch_from(self._JOBS)
        sdg = _run(lineage.nearest_ancestor(self._JOBS["eval1"], "sdg", fetch))
        assert sdg is not None and sdg["id"] == "sdg2"

    def test_nearest_ancestor_is_breadth_first(self) -> None:
        # parent_job_id (the first gated field) reaches an sdg in TWO hops; another gated
        # field reaches one DIRECTLY. "nearest" must return the one-hop match — a
        # depth-first walk would recurse the first field to completion and return the
        # deeper sdg instead.
        jobs = {
            "sdg_near": {"id": "sdg_near", "type": "sdg"},
            "sdg_far": {"id": "sdg_far", "type": "sdg"},
            "train_mid": {"id": "train_mid", "type": "training", "parent_job_id": "sdg_far"},
            "node": {
                "id": "node",
                "type": "eval",
                "parent_job_id": "train_mid",
                "config": {"training_job_id": "sdg_near"},
            },
        }
        sdg = _run(lineage.nearest_ancestor(jobs["node"], "sdg", _fetch_from(jobs)))
        assert sdg is not None and sdg["id"] == "sdg_near"

    def test_nearest_ancestor_multi_hop_through_training(self) -> None:
        # an eval chained ONLY to the model (no dataset parent) still finds the SDG
        # by walking eval → training → sdg.
        jobs = dict(self._JOBS)
        jobs["eval2"] = {"id": "eval2", "type": "eval", "config": {"training_job_id": "train1"}}
        sdg = _run(lineage.nearest_ancestor(jobs["eval2"], "sdg", _fetch_from(jobs)))
        assert sdg is not None and sdg["id"] == "sdg1"

    def test_nearest_ancestor_accepts_type_set(self) -> None:
        jobs = {
            "up1": {"id": "up1", "type": "upload", "mlflow_run_id": "run-up1"},
            "t": {"id": "t", "type": "training", "parent_job_id": "up1"},
        }
        got = _run(
            lineage.nearest_ancestor(jobs["t"], lineage.DATASET_PRODUCER_TYPES, _fetch_from(jobs))
        )
        assert got is not None and got["id"] == "up1"

    def test_ancestors_is_cycle_safe(self) -> None:
        # a self-referential parent must not loop forever
        jobs = {"t": {"id": "t", "type": "training", "parent_job_id": "t"}}
        got = _run(self._collect(jobs["t"], _fetch_from(jobs)))
        assert [j["id"] for j in got] == ["t"]  # visited once, then stops

    @staticmethod
    async def _collect(job: dict, fetch: lineage.JobFetch) -> list[dict]:
        return [a async for a in lineage.ancestors(job, fetch)]
