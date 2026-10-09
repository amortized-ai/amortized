"""Unit tests for the canonical job-dependency DAG (core/lineage.py) and its
consistency contract with the agent workflow prompts."""

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


class TestGraph:
    def test_sdg_is_a_root(self) -> None:
        assert lineage.upstreams("sdg") == ()
        assert lineage.gated_upstreams("sdg") == ()

    def test_training_depends_on_sdg(self) -> None:
        ups = lineage.upstreams("training")
        assert [u.field for u in ups] == ["parent_job_id"]
        assert ups[0].producer == "sdg"

    def test_eval_depends_on_dataset_and_model(self) -> None:
        fields = {u.field: u.producer for u in lineage.upstreams("eval")}
        assert fields == {"parent_job_id": "sdg", "training_job_id": "training"}

    def test_unknown_type_has_no_edges(self) -> None:
        assert lineage.upstreams("serve") == ()

    def test_pipeline_order_is_topological(self) -> None:
        order = lineage.pipeline_order()
        assert order == ["sdg", "training", "eval"]
        # every producer appears before its consumer
        for jt in lineage.PIPELINE:
            for up in lineage.upstreams(jt):
                assert order.index(up.producer) < order.index(jt)


class TestFieldDrivenGate:
    def test_gated_fields_match_the_dag_edges(self) -> None:
        # Anti-drift: the flat enforcement universe equals the set of gated edge
        # fields declared across PIPELINE (neither side can grow a field alone).
        dag_fields = {u.field for n in lineage.PIPELINE.values() for u in n.upstreams if u.gated}
        assert set(lineage.GATED_FIELDS) == dag_fields

    def test_present_gated_fields_picks_referenced_jobs(self) -> None:
        refs = lineage.present_gated_fields(
            {"parent_job_id": "sdg-1", "training_job_id": "", "data_run_id": "run-9", "x": 1}
        )
        # only non-empty gated job references; data_run_id is a run, not gated
        assert [(f, j) for f, j, _ in refs] == [("parent_job_id", "sdg-1")]

    def test_present_gated_fields_covers_unknown_node(self) -> None:
        # a job type "beyond the graph" still surfaces its referenced prerequisites
        refs = lineage.present_gated_fields({"training_job_id": "t-1"})
        assert [(f, j) for f, j, _ in refs] == [("training_job_id", "t-1")]


class TestDispatchDerivation:
    def test_parent_type_excludes_roots(self) -> None:
        # SDG has no prerequisite, so it is absent from the stale-reuse dispatch map.
        assert lineage.dispatch_parent_type() == {
            "validate_training_job": "training",
            "validate_eval_job": "eval",
        }

    def test_dispatch_job_type_gates_training_and_eval(self) -> None:
        assert lineage.dispatch_job_type("validate_training_job") == "training"
        assert lineage.dispatch_job_type("validate_eval_job") == "eval"

    def test_dispatch_job_type_skips_root_and_non_dispatch(self) -> None:
        assert lineage.dispatch_job_type("validate_sdg_job") is None  # root, not gated
        assert lineage.dispatch_job_type("get_job") is None
        assert lineage.dispatch_job_type("split_dataset") is None

    def test_dispatch_job_type_covers_beyond_graph(self) -> None:
        # an unknown dispatch is still gated (field-driven) on whatever it references
        assert lineage.dispatch_job_type("validate_serve_job") == "serve"


class TestWorkflowConsistency:
    def test_prompts_agree_with_the_dag(self) -> None:
        # Every gated upstream field is documented in its agent's workflow.md/skills,
        # and the orchestrator presents the stages in pipeline order.
        assert lineage.check_workflow_consistency() == []

    def test_missing_field_is_reported(self, tmp_path) -> None:
        # A prompt tree that omits training_job_id must be flagged.
        agents = tmp_path / "agents"
        (agents / "orchestrator").mkdir(parents=True)
        (agents / "orchestrator" / "workflow.md").write_text(
            "Synthetic data generation\nModel training\nModel evaluation\n"
        )
        (agents / "training").mkdir()
        (agents / "training" / "workflow.md").write_text("set parent_job_id to chain")
        (agents / "eval").mkdir()
        (agents / "eval" / "workflow.md").write_text("set parent_job_id only")  # no training_job_id
        problems = lineage.check_workflow_consistency(agents)
        assert any("training_job_id" in p for p in problems)

    def test_out_of_order_pipeline_is_reported(self, tmp_path) -> None:
        agents = tmp_path / "agents"
        (agents / "orchestrator").mkdir(parents=True)
        # evaluation presented before training — wrong order
        (agents / "orchestrator" / "workflow.md").write_text(
            "Synthetic data generation\nModel evaluation\nModel training\n"
        )
        for jt in ("training", "eval"):
            (agents / jt).mkdir()
        (agents / "training" / "workflow.md").write_text("parent_job_id")
        (agents / "eval" / "workflow.md").write_text("parent_job_id training_job_id")
        problems = lineage.check_workflow_consistency(agents)
        assert any("out of pipeline order" in p for p in problems)


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
        # eval's upstreams in declared order: parent_job_id (sdg2), then training_job_id
        # (train1), whose own parent (sdg1) is reached transitively.
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
