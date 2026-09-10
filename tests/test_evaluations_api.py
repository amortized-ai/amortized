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


class TestGrouping:
    @pytest.mark.asyncio
    async def test_groups_by_dataset_and_metric_set(self, monkeypatch) -> None:
        jobs = [
            _job(
                "j1", "base", metrics=["exact_match"],
                rubric=[{"name": "accuracy"}], eval_data_run_id="d1",
                created="2026-01-01",
            ),
            _job(
                "j2", "tuned", metrics=["exact_match"],
                rubric=[{"name": "accuracy"}], eval_data_run_id="d1",
                created="2026-01-02",
            ),
            # Same dataset, different metric set → separate group
            _job("j3", "base", metrics=["format_validity"], eval_data_run_id="d1"),
            # Different dataset
            _job(
                "j4", "base", metrics=["exact_match"],
                rubric=[{"name": "accuracy"}], eval_data_run_id="d2",
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
        groups = result["groups"]
        assert len(groups) == 3
        # The d1+exact_match+accuracy group has both models, newest first
        pair = [g for g in groups if len(g["evals"]) == 2][0]
        assert {e["model"] for e in pair["evals"]} == {"base", "tuned"}
        assert pair["evals"][0]["model"] == "tuned"  # newest first
        assert set(pair["metric_names"]) == {"exact_match", "accuracy"}
