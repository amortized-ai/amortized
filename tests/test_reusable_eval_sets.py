"""Unit tests for `list_reusable_eval_sets` (eval-set reuse lister).

DB-free: the endpoint builds `Repository(db)` internally, so we monkeypatch
that class in the `evals` module with a fake serving preset job rows, and stub
`_eval_runs_by_id` with the MLflow score tags that make an eval "scored".
"""

import pytest

from amortized.api import evals
from amortized.models import JobStatus


def _sdg_cfg(teacher: str, author_prompt: str, output_prompt: str) -> dict:
    return {
        "num_records": 100,
        "topic": "rfe",
        "model_configs": [{"alias": "teacher", "model": teacher}],
        "columns": [
            {"column_type": "sampler", "name": "topic"},
            {"column_type": "llm-text", "name": "input", "system_prompt": author_prompt},
            {"column_type": "llm-text", "name": "output", "system_prompt": output_prompt},
        ],
    }


def _eval_job(
    job_id: str,
    model: str,
    mlflow_run_id: str,
    *,
    parent: str = "",
    eval_data_run_id: str = "",
    status: str = JobStatus.succeeded.value,
) -> dict:
    cfg: dict = {
        "endpoint": {"model": model},
        "rubric": [{"name": "accuracy", "description": "is it right"}],
    }
    if eval_data_run_id:
        cfg["eval_data_run_id"] = eval_data_run_id
    return {
        "id": job_id,
        "status": status,
        "parent_job_id": parent,
        "mlflow_run_id": mlflow_run_id,
        "created_at": job_id,  # lexical order == insertion order for the tests
        "config": cfg,
    }


class _FakeRepo:
    def __init__(self, jobs_by_id: dict, eval_jobs: list[dict]) -> None:
        self._jobs = jobs_by_id
        self._eval_jobs = eval_jobs

    async def get_job(self, job_id: str):
        return self._jobs.get(job_id)

    async def list_jobs(self, job_type=None, k8s_namespace=None):
        return self._eval_jobs


@pytest.fixture
def patch_env(monkeypatch):
    def _install(jobs_by_id: dict, eval_jobs: list[dict], score_runs: dict) -> None:
        monkeypatch.setattr(
            evals, "Repository", lambda _db: _FakeRepo(jobs_by_id, eval_jobs)
        )

        async def _runs() -> dict:
            return score_runs

        monkeypatch.setattr(evals, "_eval_runs_by_id", _runs)

    return _install


def _scenario():
    """A trained model + two eval sets: one same-recipe (reusable, 2 models), one not."""
    train_sdg = {"id": "sdg-train", "config": _sdg_cfg("gpt-oss", "author", "assess RFE")}
    training = {"id": "train-1", "parent_job_id": "sdg-train"}
    # Mirror recipe -> recipe_match True; distinct prompt -> recipe_match False.
    sdg_eval_a = {"id": "sdg-eval-A", "config": _sdg_cfg("gpt-oss", "author", "assess RFE")}
    sdg_eval_b = {"id": "sdg-eval-B", "config": _sdg_cfg("gpt-oss", "author", "other task")}
    jobs_by_id = {
        "sdg-train": train_sdg,
        "train-1": training,
        "sdg-eval-A": sdg_eval_a,
        "sdg-eval-B": sdg_eval_b,
    }
    eval_jobs = [
        _eval_job("eval-A1", "tuned-v1", "run-ea1", parent="sdg-eval-A"),
        # Reuse of set A by a sibling model: only eval_data_run_id, no parent.
        _eval_job("eval-A2", "base-model", "run-ea2", eval_data_run_id="run-a"),
        _eval_job("eval-B1", "other-model", "run-eb1", parent="sdg-eval-B"),
    ]
    # sdg-eval-A's dataset run id is its mlflow_run_id; the reuse points at it.
    sdg_eval_a["mlflow_run_id"] = "run-a"
    sdg_eval_b["mlflow_run_id"] = "run-b"
    score_runs = {
        "run-ea1": {"eval_score_accuracy": "0.8"},
        "run-ea2": {"eval_score_accuracy": "0.7"},
        "run-eb1": {"eval_score_accuracy": "0.5"},
    }
    return jobs_by_id, eval_jobs, score_runs


@pytest.mark.asyncio
class TestListReusableEvalSets:
    async def test_groups_and_flags_recipe_match(self, patch_env) -> None:
        jobs_by_id, eval_jobs, score_runs = _scenario()
        patch_env(jobs_by_id, eval_jobs, score_runs)

        out = await evals.list_reusable_eval_sets("train-1", db=None)

        assert out["training_recipe_known"] is True
        assert out["recipe_match_available"] is True
        sets = {g["eval_data_run_id"]: g for g in out["eval_sets"]}
        assert set(sets) == {"run-a", "run-b"}
        # Same-recipe set comes first and carries both sibling models.
        assert out["eval_sets"][0]["eval_data_run_id"] == "run-a"
        assert sets["run-a"]["recipe_match"] is True
        assert sorted(sets["run-a"]["models_scored"]) == ["base-model", "tuned-v1"]
        assert sets["run-a"]["name"] == "rfe"  # from the generating SDG's topic
        assert sets["run-a"]["metric_names"] == ["accuracy"]
        assert sets["run-b"]["recipe_match"] is False

    async def test_skips_unscored_and_failed_evals(self, patch_env) -> None:
        jobs_by_id, eval_jobs, score_runs = _scenario()
        # A failed eval and a succeeded-but-scoreless eval on a new set must not appear.
        eval_jobs.append(
            _eval_job("eval-fail", "m", "run-f", parent="sdg-eval-A",
                      status=JobStatus.failed.value)
        )
        eval_jobs.append(_eval_job("eval-noscore", "m", "run-ns", eval_data_run_id="run-x"))
        patch_env(jobs_by_id, eval_jobs, score_runs)

        out = await evals.list_reusable_eval_sets("train-1", db=None)
        assert {g["eval_data_run_id"] for g in out["eval_sets"]} == {"run-a", "run-b"}

    async def test_recipe_unknown_when_model_not_from_sdg(self, patch_env) -> None:
        jobs_by_id, eval_jobs, score_runs = _scenario()
        jobs_by_id["train-1"] = {"id": "train-1", "parent_job_id": ""}  # no training SDG
        patch_env(jobs_by_id, eval_jobs, score_runs)

        out = await evals.list_reusable_eval_sets("train-1", db=None)
        assert out["training_recipe_known"] is False
        assert out["recipe_match_available"] is False
        # Sets are still listed so the user can choose despite no verified match.
        assert {g["eval_data_run_id"] for g in out["eval_sets"]} == {"run-a", "run-b"}
        assert all(g["recipe_match"] is False for g in out["eval_sets"])

    async def test_empty_when_no_evals(self, patch_env) -> None:
        jobs_by_id, _, _ = _scenario()
        patch_env(jobs_by_id, [], {})
        out = await evals.list_reusable_eval_sets("train-1", db=None)
        assert out["eval_sets"] == []
        assert out["recipe_match_available"] is False
