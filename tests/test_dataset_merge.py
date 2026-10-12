"""Tests for dataset merge — concatenate multiple datasets into one."""

import io
import json
import os
from unittest.mock import patch

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
    os.environ["AMORTIZED_MLFLOW_TRACKING_URI"] = "http://test"
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_conn_mod.settings = new_settings

    import amortized.api.datasets as datasets_mod

    datasets_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    import subprocess

    import asyncpg

    conn = await asyncpg.connect(TEST_DATABASE_URL)
    await conn.execute("DROP TABLE IF EXISTS alembic_version")
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.close()
    env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
    subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute("TRUNCATE jobs")
        yield c  # type: ignore[misc]


def _records(n: int, prefix: str = "q") -> list[dict]:
    return [{"messages": [{"role": "user", "content": f"{prefix}{i}"}]} for i in range(n)]


def _jsonl_bytes(records: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode()


def _parse_jsonl(raw: bytes | str) -> list[dict]:
    if isinstance(raw, bytes):
        raw = raw.decode()
    return [json.loads(line) for line in raw.strip().split("\n") if line.strip()]


def _parquet_bytes(records: list[dict]) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records), buf)  # type: ignore[no-untyped-call]
    return buf.getvalue()


class _FakeMLflow:
    """Minimal MLflow stub supporting multiple source runs for merge tests.

    ``runs`` maps run_id -> (name, [(artifact_path, raw_bytes)]).
    """

    def __init__(self, runs: dict[str, tuple[str, list[tuple[str, bytes]]]]) -> None:
        self.runs = runs
        self.uploads: dict[tuple[str, str], bytes] = {}
        self.created: list[dict] = []
        self._next = 0

    async def get_run(self, run_id: str) -> dict:
        if run_id not in self.runs:
            raise httpx.HTTPStatusError(
                "not found",
                request=httpx.Request("GET", "http://mlflow"),
                response=httpx.Response(404),
            )
        name, _ = self.runs[run_id]
        return {
            "info": {"run_id": run_id, "run_name": name},
            "data": {
                "tags": [{"key": "dataset_name", "value": name}],
            },
        }

    async def list_artifacts(self, run_id: str, path: str = "") -> list[dict]:
        _, files = self.runs[run_id]
        return [{"path": p, "file_size": len(d)} for p, d in files]

    async def get_artifact(self, run_id: str, path: str) -> bytes:
        _, files = self.runs[run_id]
        return dict(files)[path]

    async def ensure_experiment(self, name: str) -> str:
        return "exp-1"

    async def create_run(self, experiment_id: str, name: str, tags: dict) -> str:
        self._next += 1
        run_id = f"merged-run-{self._next}"
        self.created.append({"run_id": run_id, "name": name, "tags": tags})
        return run_id

    async def upload_artifact(self, run_id: str, path: str, data: bytes) -> None:
        self.uploads[(run_id, path)] = data

    async def finish_run(self, run_id: str, status: str = "FINISHED") -> None:
        pass

    async def fail_run_quiet(self, run_id: str) -> None:
        pass


async def _run_merge(
    client: httpx.AsyncClient,
    fake: _FakeMLflow,
    body: dict,
) -> dict:
    """POST the merge and wait for the background task to finish."""
    import asyncio

    with patch("amortized.api.datasets.MLflowClient", return_value=fake):
        resp = await client.post("/api/v1/datasets/merge", json=body)
        assert resp.status_code == 202, resp.text
        job = resp.json()
        for _ in range(50):
            await asyncio.sleep(0.05)
            row = await client.get(f"/api/v1/jobs/{job['id']}")
            status = row.json()["status"]
            if status in ("succeeded", "failed"):
                return row.json()
        raise AssertionError("merge job did not finish")


class TestDatasetMerge:
    async def test_basic_jsonl_merge(self, client: httpx.AsyncClient) -> None:
        a_records = _records(5, prefix="a")
        b_records = _records(3, prefix="b")
        fake = _FakeMLflow({
            "run-a": ("Dataset A", [("generated_data/data.jsonl", _jsonl_bytes(a_records))]),
            "run-b": ("Dataset B", [("generated_data/data.jsonl", _jsonl_bytes(b_records))]),
        })
        job = await _run_merge(client, fake, {"run_ids": ["run-a", "run-b"]})

        assert job["status"] == "succeeded", job.get("error")
        cfg = job["config"]
        assert cfg["num_records"] == 8
        assert cfg["merged_run_id"]

        merged = _parse_jsonl(
            fake.uploads[(cfg["merged_run_id"], "generated_data/data.jsonl")]
        )
        assert len(merged) == 8
        assert [r["messages"][0]["content"] for r in merged] == [
            "a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2"
        ]

    async def test_all_parquet_produces_parquet(self, client: httpx.AsyncClient) -> None:
        a_records = _records(4, prefix="a")
        b_records = _records(3, prefix="b")
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.parquet", _parquet_bytes(a_records))]),
            "run-b": ("B", [("generated_data/data.parquet", _parquet_bytes(b_records))]),
        })
        job = await _run_merge(client, fake, {"run_ids": ["run-a", "run-b"]})

        assert job["status"] == "succeeded"
        data = fake.uploads[(job["config"]["merged_run_id"], "generated_data/data.parquet")]
        assert data[:4] == b"PAR1"

        import pyarrow.parquet as pq

        table = pq.read_table(io.BytesIO(data))  # type: ignore[no-untyped-call]
        assert table.num_rows == 7

    async def test_mixed_formats_produce_jsonl(self, client: httpx.AsyncClient) -> None:
        """When any source is JSONL, output should be JSONL (item 1)."""
        a_records = _records(3, prefix="a")
        b_records = _records(2, prefix="b")
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.parquet", _parquet_bytes(a_records))]),
            "run-b": ("B", [("generated_data/data.jsonl", _jsonl_bytes(b_records))]),
        })
        job = await _run_merge(client, fake, {"run_ids": ["run-a", "run-b"]})

        assert job["status"] == "succeeded"
        cfg = job["config"]
        merged = _parse_jsonl(
            fake.uploads[(cfg["merged_run_id"], "generated_data/data.jsonl")]
        )
        assert len(merged) == 5

    async def test_union_schema_preserved(self, client: httpx.AsyncClient) -> None:
        """Records with different keys should all appear in the merged output."""
        a = [{"messages": [{"role": "user", "content": "a"}], "score": 0.9}]
        b = [{"messages": [{"role": "user", "content": "b"}], "metadata": {"k": "v"}}]
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.parquet", _parquet_bytes(a))]),
            "run-b": ("B", [("generated_data/data.parquet", _parquet_bytes(b))]),
        })
        job = await _run_merge(client, fake, {"run_ids": ["run-a", "run-b"]})

        assert job["status"] == "succeeded"
        import pyarrow.parquet as pq

        data = fake.uploads[(job["config"]["merged_run_id"], "generated_data/data.parquet")]
        table = pq.read_table(io.BytesIO(data))  # type: ignore[no-untyped-call]
        rows = table.to_pylist()
        assert len(rows) == 2
        assert "score" in rows[0] and "metadata" in rows[1]

    async def test_duplicate_run_ids_rejected(self, client: httpx.AsyncClient) -> None:
        """Duplicate run_ids should be rejected at the model level (item 2)."""
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.jsonl", _jsonl_bytes(_records(3)))]),
        })
        with patch("amortized.api.datasets.MLflowClient", return_value=fake):
            resp = await client.post(
                "/api/v1/datasets/merge",
                json={"run_ids": ["run-a", "run-a"]},
            )
        assert resp.status_code == 422

    async def test_empty_merge_fails(self, client: httpx.AsyncClient) -> None:
        """Merging datasets with 0 total rows should fail (item 3)."""
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.jsonl", b"\n")]),
            "run-b": ("B", [("generated_data/data.jsonl", b"\n")]),
        })
        job = await _run_merge(client, fake, {"run_ids": ["run-a", "run-b"]})

        assert job["status"] == "failed"
        assert "no rows" in job.get("error", "").lower()

    async def test_custom_name(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.jsonl", _jsonl_bytes(_records(2)))]),
            "run-b": ("B", [("generated_data/data.jsonl", _jsonl_bytes(_records(2)))]),
        })
        job = await _run_merge(
            client, fake, {"run_ids": ["run-a", "run-b"], "name": "My merged set"}
        )

        assert job["status"] == "succeeded"
        assert fake.created[0]["name"] == "My merged set"

    async def test_auto_name_filters_ds_prefix(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow({
            "run-a": ("ds-autogen", [("generated_data/data.jsonl", _jsonl_bytes(_records(2)))]),
            "run-b": ("Real Name", [("generated_data/data.jsonl", _jsonl_bytes(_records(2)))]),
        })
        job = await _run_merge(client, fake, {"run_ids": ["run-a", "run-b"]})

        assert job["status"] == "succeeded"
        assert fake.created[0]["name"] == "Real Name"

    async def test_unknown_run_404(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.jsonl", _jsonl_bytes(_records(2)))]),
        })
        with patch("amortized.api.datasets.MLflowClient", return_value=fake):
            resp = await client.post(
                "/api/v1/datasets/merge",
                json={"run_ids": ["run-a", "nonexistent"]},
            )
        assert resp.status_code == 404

    async def test_worker_does_not_pick_up_merge_jobs(self, client: httpx.AsyncClient) -> None:
        """Merge jobs are API-layer processed — the worker must not claim them."""
        import amortized.db.connection as _db_conn
        from amortized.db.repository import Repository

        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.jsonl", _jsonl_bytes(_records(3)))]),
            "run-b": ("B", [("generated_data/data.jsonl", _jsonl_bytes(_records(3)))]),
        })
        job = await _run_merge(client, fake, {"run_ids": ["run-a", "run-b"]})
        assert job["status"] == "succeeded"

        async with _db_conn._pool.acquire() as conn:
            claimed = await Repository(conn).pick_pending_job()
        assert claimed is None or claimed["id"] != job["id"]

    async def test_lineage_tags(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.jsonl", _jsonl_bytes(_records(2)))]),
            "run-b": ("B", [("generated_data/data.jsonl", _jsonl_bytes(_records(2)))]),
        })
        job = await _run_merge(client, fake, {"run_ids": ["run-a", "run-b"]})

        assert job["status"] == "succeeded"
        tags = fake.created[0]["tags"]
        assert tags["source"] == "merge"
        assert json.loads(tags["source_run_ids"]) == ["run-a", "run-b"]
        assert tags["num_samples"] == "4"

    async def test_fewer_than_two_run_ids_rejected(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow({
            "run-a": ("A", [("generated_data/data.jsonl", _jsonl_bytes(_records(2)))]),
        })
        with patch("amortized.api.datasets.MLflowClient", return_value=fake):
            resp = await client.post(
                "/api/v1/datasets/merge",
                json={"run_ids": ["run-a"]},
            )
        assert resp.status_code == 422
