"""Tests for the training-side dataset SHAPE gate (`_training_shape_violations`).

Complements the SDG-side processor gate: at the layer where the algorithm is known, a
chat-SFT run must consume `messages` and an embedding run must consume flat text/label
columns. A definitively empty dataset or a confirmed shape mismatch is a hard error; a
transient MLflow failure is best-effort (never blocks the card). The dataset read is faked,
so these are fast unit tests with no DB or MLflow.
"""

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from amortized.api import jobs as jobs_mod


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeMLflow:
    async def get_artifact(self, _run_id: str, _path: str) -> bytes:
        return b""  # content is irrelevant; _parse_records is faked


def _patch_dataset(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[dict],
    find_exc: Exception | None = None,
) -> None:
    """Fake the MLflow dataset read used by _training_shape_violations."""
    import amortized.api.datasets as ds

    async def fake_find(_mlflow: Any, _run_id: str) -> list[str]:
        if find_exc is not None:
            raise find_exc
        return ["generated_data/batch_00000.parquet"]

    monkeypatch.setattr(ds, "_mlflow_client", lambda: _FakeMLflow())
    monkeypatch.setattr(ds, "_find_dataset_artifacts", fake_find)
    monkeypatch.setattr(ds, "_parse_records", lambda _path, _data: rows)


def _violations(config: dict[str, Any]) -> list[str]:
    # data_run_id set → no Repository/DB access needed; db unused.
    return _run(jobs_mod._training_shape_violations(config, parent_job_id="", db=None))  # type: ignore[arg-type]


class TestShapeGate:
    def test_sft_with_messages_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(monkeypatch, rows=[{"messages": [{"role": "user", "content": "x"}]}])
        assert _violations({"algorithm": "sft", "data_run_id": "r"}) == []

    def test_sft_without_messages_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(monkeypatch, rows=[{"text": "x", "category": "y"}])
        errs = _violations({"algorithm": "sft", "data_run_id": "r"})
        assert len(errs) == 1 and "messages" in errs[0]

    def test_lora_sft_alias_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(monkeypatch, rows=[{"text": "x"}])
        errs = _violations({"algorithm": "qlora", "data_run_id": "r"})  # qlora -> lora_sft
        assert len(errs) == 1 and "messages" in errs[0]

    def test_embedding_with_text_label_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(monkeypatch, rows=[{"text": "x", "category": "y"}])
        assert _violations({"algorithm": "embedding_sft", "data_run_id": "r"}) == []

    def test_embedding_missing_columns_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(monkeypatch, rows=[{"messages": []}])
        errs = _violations({"algorithm": "embedding_sft", "data_run_id": "r"})
        assert len(errs) == 1 and "text" in errs[0]

    def test_embedding_custom_column_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(monkeypatch, rows=[{"prompt": "x", "label": "y"}])
        cfg = {
            "algorithm": "embedding_sft",
            "data_run_id": "r",
            "text_column": "prompt",
            "label_column": "label",
        }
        assert _violations(cfg) == []

    def test_preference_algo_not_shape_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # dpo uses a different schema; a missing `messages` must NOT be flagged.
        _patch_dataset(monkeypatch, rows=[{"prompt": "x", "chosen": "a", "rejected": "b"}])
        assert _violations({"algorithm": "dpo", "data_run_id": "r"}) == []

    def test_empty_dataset_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(
            monkeypatch, rows=[], find_exc=HTTPException(status_code=404, detail="no data")
        )
        errs = _violations({"algorithm": "sft", "data_run_id": "r"})
        assert len(errs) == 1 and "empty dataset" in errs[0]

    def test_no_rows_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(monkeypatch, rows=[])  # artifact exists but parses to zero rows
        errs = _violations({"algorithm": "sft", "data_run_id": "r"})
        assert len(errs) == 1 and "no rows" in errs[0]

    def test_transient_error_is_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(
            monkeypatch, rows=[], find_exc=HTTPException(status_code=502, detail="mlflow down")
        )
        assert _violations({"algorithm": "sft", "data_run_id": "r"}) == []

    def test_generic_error_is_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataset(monkeypatch, rows=[], find_exc=RuntimeError("boom"))
        assert _violations({"algorithm": "sft", "data_run_id": "r"}) == []

    def test_data_path_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # a direct data_path is opaque — no read attempted, no violation
        def boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("must not read dataset when data_path is set")

        import amortized.api.datasets as ds

        monkeypatch.setattr(ds, "_find_dataset_artifacts", boom)
        assert _violations({"algorithm": "sft", "data_path": "/data", "data_run_id": "r"}) == []

    def test_no_data_source_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _violations({"algorithm": "sft"}) == []
