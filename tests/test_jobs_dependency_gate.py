"""Tests for the Layer-1 fail-closed dependency gate at job creation.

Enforces the dependency invariant structurally: a downstream job cannot be created
until every upstream job it references is terminal-'succeeded'. Covers the shared
helper and both validators (notably the previously-missing eval training_job_id check).
No real DB — Repository is faked so these are fast unit tests.
"""

import asyncio
from typing import Any

import pytest

from amortized.api import jobs as jobs_mod


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


def _fake_repo_cls(store: dict[str, dict[str, Any]]) -> type:
    class FakeRepo:
        def __init__(self, _conn: Any = None) -> None:
            pass

        async def get_job(self, job_id: str) -> dict[str, Any] | None:
            return store.get(job_id)

    return FakeRepo


def _succeeded(**kw: Any) -> dict[str, Any]:
    base = {"id": "j", "status": "succeeded", "mlflow_run_id": "run123"}
    base.update(kw)
    return base


class TestRequireDependencySucceeded:
    def _repo(self, store: dict[str, dict[str, Any]]) -> Any:
        return _fake_repo_cls(store)()

    def test_missing_job(self) -> None:
        errs = _run(jobs_mod._require_dependency_succeeded(self._repo({}), "j1", "parent_job_id"))
        assert errs == ["parent_job_id: job 'j1' not found"]

    def test_in_flight_job(self) -> None:
        repo = self._repo({"j1": {"status": "running", "mlflow_run_id": "r"}})
        errs = _run(jobs_mod._require_dependency_succeeded(repo, "j1", "training_job_id"))
        assert len(errs) == 1
        assert "training_job_id" in errs[0]
        assert "'running'" in errs[0]
        assert "must finish" in errs[0]

    def test_succeeded_without_mlflow(self) -> None:
        repo = self._repo({"j1": {"status": "succeeded", "mlflow_run_id": ""}})
        errs = _run(jobs_mod._require_dependency_succeeded(repo, "j1", "parent_job_id"))
        assert len(errs) == 1
        assert "no MLflow artifacts" in errs[0]

    def test_succeeded_without_mlflow_but_not_required(self) -> None:
        repo = self._repo({"j1": {"status": "succeeded", "mlflow_run_id": ""}})
        errs = _run(
            jobs_mod._require_dependency_succeeded(
                repo, "j1", "parent_job_id", require_mlflow=False
            )
        )
        assert errs == []

    def test_fully_succeeded(self) -> None:
        repo = self._repo({"j1": _succeeded()})
        errs = _run(jobs_mod._require_dependency_succeeded(repo, "j1", "parent_job_id"))
        assert errs == []


class TestValidateEvalData:
    def test_inflight_training_job_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = {"p1": _succeeded(), "t1": {"status": "running", "mlflow_run_id": "r"}}
        monkeypatch.setattr(jobs_mod, "Repository", _fake_repo_cls(store))
        errs = _run(jobs_mod._validate_eval_data({"training_job_id": "t1"}, "p1", None))  # type: ignore[arg-type]
        assert any("training_job_id" in e and "'running'" in e for e in errs)

    def test_succeeded_training_and_parent_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = {"p1": _succeeded(), "t1": _succeeded()}
        monkeypatch.setattr(jobs_mod, "Repository", _fake_repo_cls(store))
        errs = _run(jobs_mod._validate_eval_data({"training_job_id": "t1"}, "p1", None))  # type: ignore[arg-type]
        assert errs == []

    def test_training_via_data_run_still_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No parent_job_id; dataset supplied via eval_data_run_id, but the model under
        # eval is still in flight -> training_job_id must still gate.
        store = {"t1": {"status": "queued", "mlflow_run_id": ""}}
        monkeypatch.setattr(jobs_mod, "Repository", _fake_repo_cls(store))
        cfg = {"eval_data_run_id": "a" * 32, "training_job_id": "t1"}
        errs = _run(jobs_mod._validate_eval_data(cfg, "", None))  # type: ignore[arg-type]
        assert any("training_job_id" in e for e in errs)

    def test_inflight_parent_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = {"p1": {"status": "running", "mlflow_run_id": "r"}}
        monkeypatch.setattr(jobs_mod, "Repository", _fake_repo_cls(store))
        errs = _run(jobs_mod._validate_eval_data({}, "p1", None))  # type: ignore[arg-type]
        assert any("parent_job_id" in e and "'running'" in e for e in errs)

    def test_no_data_source_errors_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(jobs_mod, "Repository", _fake_repo_cls({}))
        errs = _run(jobs_mod._validate_eval_data({}, "", None))  # type: ignore[arg-type]
        assert any("eval jobs require" in e for e in errs)


class TestValidateTrainingData:
    def test_inflight_parent_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = {"p1": {"status": "provisioning", "mlflow_run_id": "r"}}
        monkeypatch.setattr(jobs_mod, "Repository", _fake_repo_cls(store))
        errs = _run(jobs_mod._validate_training_data({}, "p1", None))  # type: ignore[arg-type]
        assert any("parent_job_id" in e and "'provisioning'" in e for e in errs)

    def test_succeeded_parent_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = {"p1": _succeeded()}
        monkeypatch.setattr(jobs_mod, "Repository", _fake_repo_cls(store))
        errs = _run(jobs_mod._validate_training_data({}, "p1", None))  # type: ignore[arg-type]
        assert errs == []

    def test_data_path_skips_parent_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A direct data_path dispatch does not chain from a parent job.
        monkeypatch.setattr(jobs_mod, "Repository", _fake_repo_cls({}))
        errs = _run(
            jobs_mod._validate_training_data({"data_path": "/data/x.jsonl"}, "p1", None)  # type: ignore[arg-type]
        )
        assert errs == []
