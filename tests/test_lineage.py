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


class TestDispatchDerivation:
    def test_parent_type_covers_training_and_eval(self) -> None:
        assert lineage.dispatch_parent_type() == {
            "validate_sdg_job": "sdg",
            "validate_training_job": "training",
            "validate_eval_job": "eval",
        }

    def test_upstream_fields_exclude_sdg(self) -> None:
        fields = lineage.dispatch_upstream_fields()
        assert "validate_sdg_job" not in fields  # SDG eval-set prep is not gated
        assert fields["validate_training_job"] == [("parent_job_id", "the training dataset's job")]
        assert [f for f, _ in fields["validate_eval_job"]] == [
            "parent_job_id",
            "training_job_id",
        ]

    def test_layers_see_the_same_edge_set(self) -> None:
        # Anti-drift: the proxy's per-tool gated fields must equal the DAG's gated
        # upstreams for the tool's job type (what Layer 1 iterates).
        fields = lineage.dispatch_upstream_fields()
        by_type = {n.validate_tool: n.job_type for n in lineage.PIPELINE.values()}
        for tool, pairs in fields.items():
            jt = by_type[tool]
            assert {f for f, _ in pairs} == {u.field for u in lineage.gated_upstreams(jt)}


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
