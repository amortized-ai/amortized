"""Tests for the /api/v1/evaluations grouping endpoint."""

import json

import pytest

from amortized.api import evals as evals_api


def _job(
    job_id: str,
    model: str,
    status: str = "succeeded",
    *,
    metrics: list | None = None,
    rubric: list | None = None,
    parent: str = "",
    run_id: str = "",
    eval_data_run_id: str = "",
    created: str = "2026-01-01T00:00:00",
) -> dict:
    return {
        "id": job_id,
        "type": "eval",
        "status": status,
        "config": {
            "endpoint": {"model": model, "base_url": "http://x/v1"},
            "metrics": metrics or [],
            "rubric": rubric or [],
            "eval_data_run_id": eval_data_run_id,
        },
        "parent_job_id": parent,
        "mlflow_run_id": run_id,
        "created_at": created,
    }


class TestMetricSetSignature:
    def test_same_metrics_same_signature(self) -> None:
        a = evals_api._metric_set_signature({"metrics": ["exact_match"], "rubric": []})
        b = evals_api._metric_set_signature({"metrics": ["exact_match"]})
        assert a == b

    def test_different_rubric_different_signature(self) -> None:
        a = evals_api._metric_set_signature(
            {"rubric": [{"name": "accuracy", "description": "d"}]}
        )
        b = evals_api._metric_set_signature(
            {"rubric": [{"name": "fluency", "description": "d"}]}
        )
        assert a != b

    def test_order_insensitive(self) -> None:
        a = evals_api._metric_set_signature({"metrics": ["a", "b"]})
        b = evals_api._metric_set_signature({"metrics": ["b", "a"]})
        assert a == b

    def test_signature_names_round_trip(self) -> None:
        sig = evals_api._metric_set_signature(
            {"metrics": ["exact_match"], "rubric": [{"name": "accuracy"}]}
        )
        names = evals_api._signature_names(sig)
        assert set(names) == {"exact_match", "accuracy"}


class TestEntryFromJob:
    def test_scores_from_tags(self) -> None:
        job = _job("j1", "m1", run_id="r1", rubric=[{"name": "accuracy"}])
        tags = {
            "r1": {
                "eval_model": "m1-tag",
                "eval_exact_match": "0.5",
                "eval_score_accuracy": "0.75",
                "num_samples": "100",
            }
        }
        entry = evals_api._entry_from_job(job, tags)
        assert entry["model"] == "m1"  # config endpoint wins
        assert entry["scores"] == {"exact_match": 0.5, "accuracy": 0.75}
        assert entry["num_samples"] == 100

    def test_model_falls_back_to_tag(self) -> None:
        job = _job("j1", "", run_id="r1")
        entry = evals_api._entry_from_job(job, {"r1": {"eval_model": "from-tag"}})
        assert entry["model"] == "from-tag"

    def test_unknown_model(self) -> None:
        entry = evals_api._entry_from_job(_job("j1", ""), {})
        assert entry["model"] == "(unknown)"

    def test_legacy_endpoint_keys(self) -> None:
        job = _job("j1", "")
        job["config"]["endpoint"] = {}
        job["config"]["endpoint_tuned"] = {"model": "legacy-tuned"}
        entry = evals_api._entry_from_job(job, {})
        assert entry["model"] == "legacy-tuned"


class TestSemanticConfig:
    def test_prefers_resolved_vllm_args_tag(self) -> None:
        # Engine-resolved args from the run tag win over raw config args.
        job = _job("j1", "m1")
        job["config"]["vllm_args"] = ["--max-model-len=32768"]
        tags = {"r-none": {}}
        cfg = evals_api._semantic_config(job["config"], {"resolved_vllm_args": json.dumps({"max_model_len": 32768})})
        assert cfg["vllm_args"] == json.dumps({"max_model_len": 32768})

    def test_resolved_tag_normalizes_explicit_default(self) -> None:
        # Explicitly-set-to-default resolves identically to unset.
        a = evals_api._semantic_config(
            {"vllm_args": ["--max-model-len=32768"]},
            {"resolved_vllm_args": json.dumps({"max_model_len": 32768})},
        )
        b = evals_api._semantic_config(
            {"vllm_args": []},
            {"resolved_vllm_args": json.dumps({"max_model_len": 32768})},
        )
        assert a["vllm_args"] == b["vllm_args"]

    def test_falls_back_to_raw_args_without_tag(self) -> None:
        job = _job("j1", "m1")
        job["config"]["vllm_args"] = ["--max-model-len=8192", "--seed=1"]
        cfg = evals_api._semantic_config(job["config"], {})
        assert json.loads(cfg["vllm_args"]) == ["--max-model-len=8192", "--seed=1"]


class TestGrouping:
    @pytest.mark.asyncio
    async def test_groups_by_dataset_and_metric_set(self, monkeypatch) -> None:
        jobs = [
            _job(
                "j1", "base", metrics=["exact_match"],
                rubric=[{"name": "accuracy"}], eval_data_run_id="d1",
                created="2026-01-01", run_id="r1",
            ),
            _job(
                "j2", "tuned", metrics=["exact_match"],
                rubric=[{"name": "accuracy"}], eval_data_run_id="d1",
                created="2026-01-02", run_id="r2",
            ),
            # Same dataset, different metric set → separate group
            _job("j3", "base", metrics=["format_validity"], eval_data_run_id="d1", run_id="r3"),
            # Different dataset
            _job(
                "j4", "base", metrics=["exact_match"],
                rubric=[{"name": "accuracy"}], eval_data_run_id="d2", run_id="r4",
            ),
        ]

        class FakeRepo:
            async def list_jobs(self, **kw):
                return jobs

            async def get_job(self, job_id):
                return None

        class FakeConn:
            pass

        async def fake_runs():
            return {
                "r1": {"eval_score_accuracy": "0.8"},
                "r2": {"eval_score_accuracy": "0.9"},
                "r3": {"eval_score_format": "0.7"},
                "r4": {"eval_score_accuracy": "0.6"},
            }

        monkeypatch.setattr(evals_api, "_eval_runs_by_id", fake_runs)
        monkeypatch.setattr(evals_api, "Repository", lambda conn: FakeRepo())
        result = await evals_api.list_evaluations(db=FakeConn())
        groups = result["groups"]
        assert len(groups) == 3
        # The d1+exact_match+accuracy group has both models, newest first
        pair = [g for g in groups if len(g["evals"]) == 2][0]
        assert {e["model"] for e in pair["evals"]} == {"base", "tuned"}
        assert pair["evals"][0]["model"] == "tuned"  # newest first
        assert set(pair["metric_names"]) == {"exact_match", "accuracy"}


class TestScoredGroupFilter:
    @pytest.mark.asyncio
    async def test_groups_without_scores_are_hidden(self, monkeypatch) -> None:
        jobs = [
            # scored eval on d1 → group kept
            _job(
                "j1", "base", metrics=["exact_match"],
                eval_data_run_id="d1", run_id="r1",
            ),
            # eval on d2 that never produced scores → group hidden
            _job("j2", "m2", metrics=["exact_match"], eval_data_run_id="d2"),
        ]

        class FakeRepo:
            async def list_jobs(self, **kw):
                return jobs

            async def get_job(self, job_id):
                return None

        class FakeConn:
            pass

        async def fake_runs():
            return {"r1": {"eval_model": "base", "eval_exact_match": "0.5"}}

        monkeypatch.setattr(evals_api, "_eval_runs_by_id", fake_runs)
        monkeypatch.setattr(evals_api, "Repository", lambda conn: FakeRepo())
        result = await evals_api.list_evaluations(db=FakeConn())
        assert len(result["groups"]) == 1
        assert result["groups"][0]["dataset"]["run_id"] == "d1"

    @pytest.mark.asyncio
    async def test_failed_evals_excluded_from_group(self, monkeypatch) -> None:
        # A failed eval on the same dataset + metric set as succeeded ones
        # must not appear as a (scoreless) column in the comparison table.
        jobs = [
            _job(
                "j1", "base", metrics=["exact_match"],
                eval_data_run_id="d1", run_id="r1", created="2026-01-01",
            ),
            _job(
                "j2", "broken", status="failed",
                metrics=["exact_match"], eval_data_run_id="d1", run_id="r2",
                created="2026-01-02",
            ),
            _job(
                "j3", "tuned", metrics=["exact_match"],
                eval_data_run_id="d1", run_id="r3", created="2026-01-03",
            ),
        ]

        class FakeRepo:
            async def list_jobs(self, **kw):
                return jobs

            async def get_job(self, job_id):
                return None

        class FakeConn:
            pass

        async def fake_runs():
            return {
                "r1": {"eval_score_exact_match": "0.5"},
                "r3": {"eval_score_exact_match": "0.6"},
            }

        monkeypatch.setattr(evals_api, "_eval_runs_by_id", fake_runs)
        monkeypatch.setattr(evals_api, "Repository", lambda conn: FakeRepo())
        result = await evals_api.list_evaluations(db=FakeConn())
        assert len(result["groups"]) == 1
        models = [e["model"] for e in result["groups"][0]["evals"]]
        assert models == ["tuned", "base"]  # newest first, no "broken"

    @pytest.mark.asyncio
    async def test_failed_only_group_hidden(self, monkeypatch) -> None:
        jobs = [
            _job(
                "j1", "m1", status="failed",
                metrics=["exact_match"], eval_data_run_id="d1",
            ),
        ]

        class FakeRepo:
            async def list_jobs(self, **kw):
                return jobs

            async def get_job(self, job_id):
                return None

        class FakeConn:
            pass

        async def fake_runs():
            return {}

        monkeypatch.setattr(evals_api, "_eval_runs_by_id", fake_runs)
        monkeypatch.setattr(evals_api, "Repository", lambda conn: FakeRepo())
        result = await evals_api.list_evaluations(db=FakeConn())
        assert result["groups"] == []
