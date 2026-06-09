"""Tests for core/events.py — no HTTP server required."""

import aiosqlite
import pytest

from amortized.core.events import Event, emit_event, list_events
from amortized.db.connection import _SCHEMA_PATH
from amortized.db.repository import Repository
from amortized.models import JobType


@pytest.fixture
async def repo(tmp_path):
    db_path = tmp_path / "test.db"
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    await db.executescript(_SCHEMA_PATH.read_text())
    await db.commit()
    repo = Repository(db)
    # Seed a job for FK constraints
    await repo.create_job(
        job_id="j1",
        job_type=JobType.training,
        config={},
        created_at="2026-01-01T00:00:00",
    )
    yield repo
    await db.close()


class TestEmitEvent:
    @pytest.mark.asyncio
    async def test_emit_creates_event(self, repo: Repository) -> None:
        event = await emit_event(repo, "j1", "state_change", {"status": "pending"})
        assert isinstance(event, Event)
        assert event.job_id == "j1"
        assert event.type == "state_change"
        assert event.data == {"status": "pending"}
        assert event.id
        assert event.timestamp

    @pytest.mark.asyncio
    async def test_emit_persists_to_db(self, repo: Repository) -> None:
        await emit_event(repo, "j1", "state_change", {"status": "running"})
        events = await list_events(repo, "j1")
        assert len(events) == 1
        assert events[0]["type"] == "state_change"
        assert events[0]["data"]["status"] == "running"


class TestListEvents:
    @pytest.mark.asyncio
    async def test_list_empty(self, repo: Repository) -> None:
        events = await list_events(repo, "j1")
        assert events == []

    @pytest.mark.asyncio
    async def test_list_multiple(self, repo: Repository) -> None:
        await emit_event(repo, "j1", "state_change", {"status": "pending"})
        await emit_event(repo, "j1", "state_change", {"status": "running"})
        events = await list_events(repo, "j1")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_list_filters_by_job_id(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j2",
            job_type=JobType.sdg,
            config={},
            created_at="2026-01-01T00:00:01",
        )
        await emit_event(repo, "j1", "state_change")
        await emit_event(repo, "j2", "state_change")
        assert len(await list_events(repo, "j1")) == 1
        assert len(await list_events(repo, "j2")) == 1
