"""Tests for dataset upload endpoint."""

import io
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from conftest import TEST_DATABASE_URL

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db.connection as db_conn_mod

    os.environ["AMORTIZED_DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


@pytest.mark.asyncio
async def test_upload_dataset_jsonl(client: httpx.AsyncClient) -> None:
    content = b'{"messages": [{"role": "user", "content": "hi"}]}\n'
    with patch(
        "amortized.api.datasets._store_dataset_in_mlflow", new_callable=AsyncMock
    ) as mock_mlflow:
        mock_mlflow.return_value = ("fake-run-id", "fake-experiment-id")
        resp = await client.post(
            "/api/v1/datasets/upload",
            files={"file": ("train.jsonl", io.BytesIO(content), "application/octet-stream")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "upload"
    assert data["status"] == "succeeded"
    assert data["mlflow_run_id"] == "fake-run-id"
    assert data["config"]["source"] == "upload"
    assert data["config"]["original_filename"] == "train.jsonl"


@pytest.mark.asyncio
async def test_upload_dataset_rejects_csv(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/datasets/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    assert resp.status_code == 400
    msg = resp.json()["message"].lower()
    assert "jsonl" in msg or "parquet" in msg


@pytest.mark.asyncio
async def test_upload_dataset_rejects_empty(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/datasets/upload",
        files={"file": ("empty.jsonl", io.BytesIO(b""), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["message"].lower()
