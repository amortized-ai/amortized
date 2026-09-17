"""Tests for dataset splitting — materialized portion + complement."""

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

    # datasets.py binds settings at import time — repoint it at the test settings
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


def _records(n: int) -> list[dict]:
    return [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(n)]


def _jsonl_bytes(records: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode()


def _parse_jsonl(raw: bytes | str) -> list[dict]:
    if isinstance(raw, bytes):
        raw = raw.decode()
    return [json.loads(line) for line in raw.strip().split("\n") if line.strip()]


class _FakeMLflow:
    """Just enough MLflow for the split flow: get_run, list_artifacts,
    get_artifact, ensure_experiment, create_run, upload_artifact, finish_run.

    ``files`` is a list of (artifact_path, raw_bytes) — a single
    generated_data/data.jsonl for uploads, or multiple batch files for
    SDG datasets.
    """

    def __init__(
        self,
        records: list[dict],
        name: str = "source dataset",
        fmt: str = "jsonl",
        raw: bytes | None = None,
        files: list[tuple[str, bytes]] | None = None,
    ) -> None:
        self.name = name
        if files is None:
            if raw is None:
                raw = _jsonl_bytes(records)
            files = [(f"generated_data/data.{fmt}", raw)]
        self.files = dict(files)
        self.uploads: dict[str, bytes] = {}
        self.created: list[dict] = []
        self._next = 0

    async def get_run(self, run_id: str) -> dict:
        return {
            "info": {"run_id": run_id, "run_name": self.name},
            "data": {
                "tags": [
                    {"key": "dataset_name", "value": self.name},
                    {"key": "dataset_topic", "value": "rfe"},
                ],
            },
        }

    async def list_artifacts(self, run_id: str, path: str = "") -> list[dict]:
        return [{"path": p, "file_size": 1} for p in self.files]

    async def get_artifact(self, run_id: str, path: str) -> bytes:
        return self.files[path]

    async def ensure_experiment(self, name: str) -> str:
        return "exp-1"

    async def create_run(self, experiment_id: str, name: str, tags: dict) -> str:
        self._next += 1
        run_id = f"split-run-{self._next}"
        self.created.append({"run_id": run_id, "name": name, "tags": tags})
        return run_id

    async def upload_artifact(self, run_id: str, path: str, data: bytes) -> None:
        self.uploads[(run_id, path)] = data

    async def finish_run(self, run_id: str, status: str = "FINISHED") -> None:
        pass

    async def fail_run_quiet(self, run_id: str) -> None:
        pass


async def _run_split(
    client: httpx.AsyncClient,
    fake: _FakeMLflow,
    body: dict,
) -> dict:
    """POST the split and wait for the background task to finish."""
    import asyncio

    with patch("amortized.api.datasets.MLflowClient", return_value=fake):
        resp = await client.post("/api/v1/datasets/source-run/split", json=body)
        assert resp.status_code == 202, resp.text
        job = resp.json()
        # the background task runs on the loop; give it a moment
        for _ in range(50):
            await asyncio.sleep(0.05)
            row = await client.get(f"/api/v1/jobs/{job['id']}")
            status = row.json()["status"]
            if status in ("succeeded", "failed"):
                return row.json()
        raise AssertionError("split job did not finish")


class TestDatasetSplit:
    async def test_fraction_split_with_complement(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow(_records(10))
        job = await _run_split(client, fake, {"fraction": 0.2, "seed": 7})

        assert job["status"] == "succeeded", job.get("error")
        cfg = job["config"]
        assert cfg["num_portion"] == 2
        assert cfg["num_complement"] == 8
        assert cfg["split_run_id"] and cfg["complement_run_id"]
        assert cfg["split_run_id"] != cfg["complement_run_id"]

        # two materialized runs, disjoint rows, format preserved
        assert len(fake.created) == 2
        portion = _parse_jsonl(fake.uploads[(cfg["split_run_id"], "generated_data/data.jsonl")])
        complement = _parse_jsonl(
            fake.uploads[(cfg["complement_run_id"], "generated_data/data.jsonl")]
        )
        assert len(portion) == 2 and len(complement) == 8
        assert {r["messages"][0]["content"] for r in portion}.isdisjoint(
            {r["messages"][0]["content"] for r in complement}
        )
        # lineage tags
        portion_run = next(r for r in fake.created if r["run_id"] == cfg["split_run_id"])
        tags = {k: v for k, v in portion_run["tags"].items() if k != "split"}
        assert tags["source"] == "split"
        assert tags["source_run_id"] == "source-run"
        assert tags["num_samples"] == "2"
        assert tags["dataset_topic"] == "rfe"

    async def test_random_split_is_deterministic(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow(_records(20))
        first = await _run_split(client, fake, {"count": 5, "seed": 3})
        fake2 = _FakeMLflow(_records(20))
        second = await _run_split(client, fake2, {"count": 5, "seed": 3})

        a = _parse_jsonl(
            fake.uploads[(first["config"]["split_run_id"], "generated_data/data.jsonl")]
        )
        b = _parse_jsonl(
            fake2.uploads[(second["config"]["split_run_id"], "generated_data/data.jsonl")]
        )
        assert a == b

    async def test_head_and_tail_strategies(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow(_records(10))
        head = await _run_split(
            client, fake, {"count": 3, "strategy": "head", "create_complement": False}
        )
        portion = _parse_jsonl(
            fake.uploads[(head["config"]["split_run_id"], "generated_data/data.jsonl")]
        )
        assert [r["messages"][0]["content"] for r in portion] == ["q0", "q1", "q2"]

        fake2 = _FakeMLflow(_records(10))
        tail = await _run_split(
            client, fake2, {"count": 3, "strategy": "tail", "create_complement": False}
        )
        portion = _parse_jsonl(
            fake2.uploads[(tail["config"]["split_run_id"], "generated_data/data.jsonl")]
        )
        assert [r["messages"][0]["content"] for r in portion] == ["q7", "q8", "q9"]

    async def test_no_complement(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow(_records(10))
        job = await _run_split(client, fake, {"count": 4, "create_complement": False})
        assert job["config"]["complement_run_id"] == ""
        assert len(fake.created) == 1
        assert (
            len(
                _parse_jsonl(
                    fake.uploads[(job["config"]["split_run_id"], "generated_data/data.jsonl")]
                )
            )
            == 4
        )

    async def test_parquet_format_preserved(self, client: httpx.AsyncClient) -> None:
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq

        records = _records(10)
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(records), buf)  # type: ignore[no-untyped-call]
        fake = _FakeMLflow(records, fmt="parquet", raw=buf.getvalue())
        job = await _run_split(client, fake, {"count": 4, "create_complement": False})

        data = fake.uploads[(job["config"]["split_run_id"], "generated_data/data.parquet")]
        assert data[:4] == b"PAR1"
        table = pq.read_table(io.BytesIO(data))  # type: ignore[no-untyped-call]
        portion = table.to_pylist()
        assert len(portion) == 4
        assert {r["messages"][0]["content"] for r in portion} <= {
            r["messages"][0]["content"] for r in records
        }

    async def test_multi_batch_dataset_split_all_files(self, client: httpx.AsyncClient) -> None:
        """SDG datasets upload multiple batch files — the split must cover
        every file, not just the first one."""
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq

        # two parquet batches of 1000 rows each, like an SDG 2k dataset
        files = []
        for b in range(2):
            records = [
                {"messages": [{"role": "user", "content": f"b{b}-q{i}"}]} for i in range(1000)
            ]
            buf = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(records), buf)  # type: ignore[no-untyped-call]
            files.append((f"generated_data/batch_{b:05d}.parquet", buf.getvalue()))
        fake = _FakeMLflow([], files=files)

        job = await _run_split(client, fake, {"fraction": 0.5, "seed": 42})
        assert job["status"] == "succeeded", job.get("error")
        cfg = job["config"]
        # 2000 total — not the 1000 the single-artifact bug produced
        assert cfg["num_portion"] == 1000
        assert cfg["num_complement"] == 1000

        def _read(run_id: str, fname: str) -> set[str]:
            data = fake.uploads[(run_id, f"generated_data/{fname}")]
            assert data[:4] == b"PAR1"
            table = pq.read_table(io.BytesIO(data))  # type: ignore[no-untyped-call]
            return {r["messages"][0]["content"] for r in table.to_pylist()}

        # both batch files present in each output, batch layout preserved
        portion = _read(cfg["split_run_id"], "batch_00000.parquet") | _read(
            cfg["split_run_id"], "batch_00001.parquet"
        )
        complement = _read(cfg["complement_run_id"], "batch_00000.parquet") | _read(
            cfg["complement_run_id"], "batch_00001.parquet"
        )
        assert len(portion) == 1000 and len(complement) == 1000
        assert portion.isdisjoint(complement)
        assert portion | complement == {f"b{b}-q{i}" for b in range(2) for i in range(1000)}

    async def test_multi_batch_jsonl_count_across_files(self, client: httpx.AsyncClient) -> None:
        files = [
            ("generated_data/batch_00000.jsonl", _jsonl_bytes(_records(10))),
            ("generated_data/batch_00001.jsonl", _jsonl_bytes(_records(30))),
        ]
        fake = _FakeMLflow([], files=files)
        job = await _run_split(client, fake, {"count": 8, "create_complement": False})

        assert job["config"]["num_portion"] == 8
        paths = [p for (rid, p) in fake.uploads if rid == job["config"]["split_run_id"]]
        assert paths == ["generated_data/batch_00000.jsonl", "generated_data/batch_00001.jsonl"]
        portion = []
        for p in paths:
            portion += _parse_jsonl(fake.uploads[(job["config"]["split_run_id"], p)])
        assert len(portion) == 8

    async def test_count_and_fraction_are_exclusive(self, client: httpx.AsyncClient) -> None:
        fake = _FakeMLflow(_records(10))
        with patch("amortized.api.datasets.MLflowClient", return_value=fake):
            resp = await client.post(
                "/api/v1/datasets/source-run/split",
                json={"count": 3, "fraction": 0.3},
            )
        assert resp.status_code == 422

    async def test_unknown_run_404(self, client: httpx.AsyncClient) -> None:
        async def _boom(run_id: str) -> dict:
            import httpx as _httpx

            raise _httpx.HTTPStatusError(
                "not found",
                request=_httpx.Request("GET", "http://mlflow"),
                response=_httpx.Response(404),
            )

        with patch("amortized.api.datasets.MLflowClient") as mock_cls:
            mock_cls.return_value.get_run = _boom
            resp = await client.post("/api/v1/datasets/nope/split", json={"count": 3})
        assert resp.status_code == 404

    async def test_worker_does_not_pick_up_split_jobs(self, client: httpx.AsyncClient) -> None:
        """Split jobs are API-layer processed — the worker must not claim them."""
        import amortized.db.connection as _db_conn
        from amortized.db.repository import Repository

        fake = _FakeMLflow(_records(10))
        job = await _run_split(client, fake, {"fraction": 0.5})
        assert job["status"] == "succeeded"

        async with _db_conn._pool.acquire() as conn:
            claimed = await Repository(conn).pick_pending_job()
        assert claimed is None or claimed["id"] != job["id"]
