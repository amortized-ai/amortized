"""Unit tests for the canonical job-dependency DAG (core/lineage.py) and its
consistency contract with the agent workflow prompts."""

from amortized.core import lineage


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
