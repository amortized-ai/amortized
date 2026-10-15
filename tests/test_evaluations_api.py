"""Tests for the /api/v1/evaluations grouping endpoint."""


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
                "eval_score_accuracy": "0.75",
                "eval_score_completeness": "0.6",
                "num_samples": "100",
            }
        }
        entry = evals_api._entry_from_job(job, tags)
        assert entry["model"] == "m1"  # config endpoint wins
        assert entry["scores"] == {"accuracy": 0.75, "completeness": 0.6}
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

    def test_classification_base_model_from_config(self) -> None:
        # A base-model classification eval (no endpoint, no eval_model tag) must
        # report its model_name_or_path, not "(unknown)".
        job = _job("j1", "")
        job["config"]["endpoint"] = {}
        job["config"]["eval_mode"] = "classification"
        job["config"]["model_name_or_path"] = "sentence-transformers/all-MiniLM-L6-v2"
        entry = evals_api._entry_from_job(job, {})
        assert entry["model"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_classification_tuned_model_label(self) -> None:
        # A tuned classification eval resolves to the training-model label (so it
        # doesn't collide with the base run and merge into one column).
        job = _job("j2", "")
        job["config"]["endpoint"] = {}
        job["config"]["eval_mode"] = "classification"
        job["config"]["training_job_id"] = "abcd1234-5678"
        labels = {"abcd1234-5678": "all-MiniLM-L6-v2-embedding_sft-abcd1234"}
        entry = evals_api._entry_from_job(job, {}, labels)
        assert entry["model"] == "all-MiniLM-L6-v2-embedding_sft-abcd1234"

    def test_classification_tuned_without_label_uses_short_id(self) -> None:
        job = _job("j3", "")
        job["config"]["endpoint"] = {}
        job["config"]["eval_mode"] = "classification"
        job["config"]["training_job_id"] = "abcd1234-5678"
        entry = evals_api._entry_from_job(job, {})
        assert entry["model"] == "tuned-abcd1234"

    def test_classification_base_and_tuned_do_not_merge_label(self) -> None:
        # The two runs must carry distinct model labels so the UI renders them as
        # separate columns instead of a merged mean +/- std.
        base = _job("jb", "")
        base["config"].update(
            endpoint={}, eval_mode="classification",
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        )
        tuned = _job("jt", "")
        tuned["config"].update(
            endpoint={}, eval_mode="classification", training_job_id="abcd1234-5678",
        )
        labels = {"abcd1234-5678": "all-MiniLM-L6-v2-embedding_sft-abcd1234"}
        mb = evals_api._entry_from_job(base, {})["model"]
        mt = evals_api._entry_from_job(tuned, {}, labels)["model"]
        assert mb != mt and mb != "(unknown)" and mt != "(unknown)"


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
        pair = next(g for g in groups if len(g["evals"]) == 2)
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
            return {"r1": {"eval_model": "base", "eval_score_accuracy": "0.5"}}

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
