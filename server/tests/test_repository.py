"""Tests for the Repository CRUD class — no HTTP server required."""

import aiosqlite
import pytest

from amortized.db.connection import _SCHEMA_PATH
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType


@pytest.fixture
async def repo(tmp_path):
    db_path = tmp_path / "test.db"
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    schema_sql = _SCHEMA_PATH.read_text()
    await db.executescript(schema_sql)
    await db.commit()
    repo = Repository(db)
    yield repo
    await db.close()


class TestJobCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get_job(self, repo: Repository) -> None:
        row = await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={"model_name_or_path": "test"},
            created_at="2026-01-01T00:00:00",
        )
        assert row["id"] == "j1"
        assert row["type"] == "training"
        assert row["status"] == "validating"
        assert row["config"]["model_name_or_path"] == "test"

        fetched = await repo.get_job("j1")
        assert fetched is not None
        assert fetched["id"] == "j1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, repo: Repository) -> None:
        assert await repo.get_job("nope") is None

    @pytest.mark.asyncio
    async def test_list_jobs_with_filters(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        await repo.create_job(
            job_id="j2",
            job_type=JobType.sdg,
            config={},
            created_at="2026-01-01T00:00:01",
        )

        all_jobs = await repo.list_jobs()
        assert len(all_jobs) == 2

        training_only = await repo.list_jobs(job_type=JobType.training)
        assert len(training_only) == 1
        assert training_only[0]["type"] == "training"

    @pytest.mark.asyncio
    async def test_update_job_status(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        updated = await repo.update_job_status(
            "j1",
            status=JobStatus.running,
            updated_at="2026-01-01T00:01:00",
            started_at="2026-01-01T00:01:00",
            pid=12345,
        )
        assert updated is not None
        assert updated["status"] == "running"
        assert updated["pid"] == 12345

    @pytest.mark.asyncio
    async def test_error_field_null_not_string_none(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        job = await repo.get_job("j1")
        assert job is not None
        assert job["error"] is None

    @pytest.mark.asyncio
    async def test_error_string_none_sanitized(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        await repo.conn.execute("UPDATE jobs SET error = 'None' WHERE id = 'j1'")
        await repo.conn.commit()
        job = await repo.get_job("j1")
        assert job is not None
        assert job["error"] is None


class TestArtifactCRUD:
    @pytest.mark.asyncio
    async def test_create_and_list_artifacts(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        await repo.create_artifact(
            artifact_id="a1",
            job_id="j1",
            artifact_type="adapter_config",
            path="/out/adapter_config.json",
            size=100,
            created_at="2026-01-01T00:01:00",
        )
        artifacts = await repo.list_artifacts("j1")
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_type"] == "adapter_config"

    @pytest.mark.asyncio
    async def test_get_artifact(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        await repo.create_artifact(
            artifact_id="a1",
            job_id="j1",
            artifact_type="log",
            path="/out/stdout.log",
            size=50,
            created_at="2026-01-01T00:01:00",
        )
        a = await repo.get_artifact("a1")
        assert a is not None
        assert a["path"] == "/out/stdout.log"

        assert await repo.get_artifact("nope") is None


class TestEventCRUD:
    @pytest.mark.asyncio
    async def test_create_and_list_events(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        await repo.create_event(
            event_id="e1",
            job_id="j1",
            event_type="state_change",
            timestamp="2026-01-01T00:00:01",
            data={"status": "pending"},
        )
        await repo.create_event(
            event_id="e2",
            job_id="j1",
            event_type="state_change",
            timestamp="2026-01-01T00:00:02",
            data={"status": "running"},
        )
        events = await repo.list_events("j1")
        assert len(events) == 2
        assert events[0]["type"] == "state_change"
        assert events[0]["data"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_event_with_no_data(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        event = await repo.create_event(
            event_id="e1",
            job_id="j1",
            event_type="heartbeat",
            timestamp="2026-01-01T00:00:01",
        )
        assert event["data"] is None


class TestConversationCRUD:
    @pytest.mark.asyncio
    async def test_create_and_list_conversations(self, repo: Repository) -> None:
        await repo.create_conversation(
            conversation_id="c1",
            title="Test",
            created_at="2026-01-01T00:00:00",
        )
        convos = await repo.list_conversations()
        assert len(convos) == 1
        assert convos[0]["title"] == "Test"

    @pytest.mark.asyncio
    async def test_create_and_list_messages(self, repo: Repository) -> None:
        await repo.create_conversation(
            conversation_id="c1",
            title="Test",
            created_at="2026-01-01T00:00:00",
        )
        await repo.create_message(
            message_id="m1",
            conversation_id="c1",
            role="user",
            content='{"text": "hello"}',
            created_at="2026-01-01T00:00:01",
        )
        msgs = await repo.list_messages("c1")
        assert len(msgs) == 1
        assert msgs[0]["content"]["text"] == "hello"
