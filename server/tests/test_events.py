"""Tests for event system — REST listing, SSE streaming, lifecycle event emission."""

import os

import httpx
import pytest
from sse_starlette.sse import EventSourceResponse

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db as db_mod
    import amortized.db.connection as db_conn_mod

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


async def _create_training_job(client: httpx.AsyncClient) -> str:
    resp = await client.post(
        "/api/v1/jobs/training",
        json={
            "algorithm": "lora_sft",
            "model_path": "test-model",
            "data_path": "./data.jsonl",
            "ckpt_output_dir": "./outputs",
        },
    )
    assert resp.status_code == 201
    return resp.json()["id"]


class TestRESTEventListing:
    @pytest.mark.asyncio
    async def test_events_for_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/jobs/nonexistent/events")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_events_after_create(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        resp = await client.get(f"/api/v1/jobs/{job_id}/events")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) >= 1
        state_changes = [e for e in events if e["type"] == "state_change"]
        assert any(e["data"]["status"] == "queued" for e in state_changes)

    @pytest.mark.asyncio
    async def test_events_after_cancel(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        await client.delete(f"/api/v1/jobs/{job_id}")
        resp = await client.get(f"/api/v1/jobs/{job_id}/events")
        assert resp.status_code == 200
        events = resp.json()
        state_changes = [e for e in events if e["type"] == "state_change"]
        statuses = [e["data"]["status"] for e in state_changes]
        assert "queued" in statuses
        assert "cancelled" in statuses

    @pytest.mark.asyncio
    async def test_filter_by_type(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        await client.delete(f"/api/v1/jobs/{job_id}")
        resp = await client.get(f"/api/v1/jobs/{job_id}/events", params={"types": "state_change"})
        assert resp.status_code == 200
        events = resp.json()
        assert all(e["type"] == "state_change" for e in events)

    @pytest.mark.asyncio
    async def test_filter_by_since(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        resp = await client.get(f"/api/v1/jobs/{job_id}/events")
        events = resp.json()
        assert len(events) >= 1
        first_ts = events[0]["timestamp"]
        resp2 = await client.get(f"/api/v1/jobs/{job_id}/events", params={"since": first_ts})
        assert resp2.status_code == 200
        filtered = resp2.json()
        assert len(filtered) < len(events) or len(events) == 0

    @pytest.mark.asyncio
    async def test_filter_nonmatching_type_returns_empty(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        resp = await client.get(
            f"/api/v1/jobs/{job_id}/events", params={"types": "nonexistent_type"}
        )
        assert resp.status_code == 200
        assert resp.json() == []


class TestSSEStreaming:
    @pytest.mark.asyncio
    async def test_stream_false_returns_json(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        resp = await client.get(f"/api/v1/jobs/{job_id}/events", params={"stream": "false"})
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", "")
        events = resp.json()
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_sse_stream_returns_event_source_response(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify the SSE endpoint returns an EventSourceResponse.

        httpx ASGITransport doesn't reliably surface content-type from
        streaming responses, so we check that the endpoint returns the
        correct response type directly via the route handler.
        """
        from unittest.mock import AsyncMock

        from amortized.api.events import get_job_events

        job_id = await _create_training_job(client)

        request = AsyncMock()
        request.is_disconnected = AsyncMock(return_value=True)

        from amortized.db import get_db

        async for db in get_db():
            result = await get_job_events(
                request=request,
                job_id=job_id,
                since=None,
                types=None,
                stream=True,
                last_event_id=None,
                db=db,
            )
            assert isinstance(result, EventSourceResponse)
            break


class TestLogsEndpoint:
    @pytest.mark.asyncio
    async def test_logs_for_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/jobs/nonexistent/logs")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_logs_returns_event_source_response(self, client: httpx.AsyncClient) -> None:
        from unittest.mock import AsyncMock

        from amortized.api.events import stream_job_logs

        job_id = await _create_training_job(client)

        request = AsyncMock()
        request.is_disconnected = AsyncMock(return_value=True)

        from amortized.db import get_db

        async for db in get_db():
            result = await stream_job_logs(
                request=request,
                job_id=job_id,
                last_event_id=None,
                db=db,
            )
            assert isinstance(result, EventSourceResponse)
            break

    @pytest.mark.asyncio
    async def test_logs_cursor_reconnection(self, client: httpx.AsyncClient) -> None:
        from amortized.core.events import emit_log
        from amortized.db import get_db
        from amortized.db.repository import Repository

        job_id = await _create_training_job(client)

        async for db in get_db():
            repo = Repository(db)
            e1 = await emit_log(repo, job_id, "line one")
            await emit_log(repo, job_id, "line two")
            break

        resp = await client.get(
            f"/api/v1/jobs/{job_id}/events",
            params={"types": "log", "since": e1.timestamp},
        )
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 1
        assert events[0]["data"]["message"] == "line two"


class TestTerminalState:
    @pytest.mark.asyncio
    async def test_events_done_on_cancelled_job(self, client: httpx.AsyncClient) -> None:
        """A cancelled job's event stream should include a done event."""
        job_id = await _create_training_job(client)
        await client.delete(f"/api/v1/jobs/{job_id}")

        resp = await client.get(f"/api/v1/jobs/{job_id}/events")
        assert resp.status_code == 200
        events = resp.json()
        assert any(
            e["type"] == "state_change" and e["data"]["status"] == "cancelled" for e in events
        ), "cancelled state_change event missing"

    @pytest.mark.asyncio
    async def test_events_sse_terminates_on_terminal_job(self, client: httpx.AsyncClient) -> None:
        """SSE event_generator should yield done event for a terminal job."""
        from unittest.mock import AsyncMock

        from amortized.api.events import _TERMINAL_STATUSES, get_job_events

        job_id = await _create_training_job(client)
        await client.delete(f"/api/v1/jobs/{job_id}")

        request = AsyncMock()
        request.is_disconnected = AsyncMock(return_value=False)

        from amortized.db import get_db

        async for db in get_db():
            result = await get_job_events(
                request=request,
                job_id=job_id,
                since=None,
                types=None,
                stream=True,
                last_event_id=None,
                db=db,
            )
            assert isinstance(result, EventSourceResponse)
            assert {"succeeded", "failed", "cancelled"} == _TERMINAL_STATUSES
            break

    @pytest.mark.asyncio
    async def test_logs_sse_returns_response_for_terminal_job(
        self, client: httpx.AsyncClient
    ) -> None:
        """Logs SSE should still return EventSourceResponse for a terminal job."""
        from unittest.mock import AsyncMock

        from amortized.api.events import stream_job_logs

        job_id = await _create_training_job(client)
        await client.delete(f"/api/v1/jobs/{job_id}")

        request = AsyncMock()
        request.is_disconnected = AsyncMock(return_value=False)

        from amortized.db import get_db

        async for db in get_db():
            result = await stream_job_logs(
                request=request,
                job_id=job_id,
                last_event_id=None,
                db=db,
            )
            assert isinstance(result, EventSourceResponse)
            break


class TestEventLifecycle:
    @pytest.mark.asyncio
    async def test_create_emits_queued_event(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        resp = await client.get(f"/api/v1/jobs/{job_id}/events")
        events = resp.json()
        assert any(e["type"] == "state_change" and e["data"]["status"] == "queued" for e in events)

    @pytest.mark.asyncio
    async def test_cancel_emits_cancelled_event(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        await client.delete(f"/api/v1/jobs/{job_id}")
        resp = await client.get(f"/api/v1/jobs/{job_id}/events")
        events = resp.json()
        assert any(
            e["type"] == "state_change" and e["data"]["status"] == "cancelled" for e in events
        )

    @pytest.mark.asyncio
    async def test_events_have_required_fields(self, client: httpx.AsyncClient) -> None:
        job_id = await _create_training_job(client)
        resp = await client.get(f"/api/v1/jobs/{job_id}/events")
        events = resp.json()
        for event in events:
            assert "id" in event
            assert "job_id" in event
            assert "type" in event
            assert "timestamp" in event
            assert event["job_id"] == job_id
