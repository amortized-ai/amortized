"""Job admission: model checks gate validate/create/retry through one shared path,
without holding a DB connection while probing."""

import asyncio
import json
from typing import Any

import httpx
import pytest
import test_eval_jobs
from test_eval_jobs import EVAL_BODY

import amortized.config as config_mod
from amortized.api import job_admission
from amortized.core import model_check_client
from amortized.core.jobs import InvalidJobStateError, create_job
from amortized.core.model_check_client import CheckReport
from amortized.core.model_endpoints import FAILED, UNVERIFIED, CheckResult
from amortized.jobs import eval as eval_builder
from amortized.model_check import checker
from amortized.models import JobType

OPENAI_OK = {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}

# Reuse the eval tests' DB + ASGI client fixtures.
_use_temp_db = test_eval_jobs._use_temp_db
client = test_eval_jobs.client


async def _job_count() -> int:
    import amortized.db.connection as db

    async with db._pool.acquire() as conn:
        return int(await conn.fetchval("SELECT count(*) FROM jobs"))


def _stub_report(monkeypatch: pytest.MonkeyPatch, report: CheckReport) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def fake(job_type: str, config: dict[str, Any]) -> CheckReport:
        seen.append({"job_type": job_type, "config": json.loads(json.dumps(config))})
        return report

    monkeypatch.setattr(job_admission, "check_job_models", fake)
    return seen


@pytest.fixture
def real_check(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Run the real in-process check against a mock network instead of the offline stub
    (switched on when the test installs its handler, so setup can still create jobs)."""
    monkeypatch.setattr(config_mod.settings, "model_check_url", "")
    monkeypatch.setattr(eval_builder, "resolve_provider", lambda _n: None)
    monkeypatch.setattr(checker, "BACKOFF_SECONDS", (0.0, 0.0))
    real_client = httpx.AsyncClient

    def install(handler) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(job_admission, "check_job_models", model_check_client.check_job_models)

        async def _handle(request: httpx.Request) -> httpx.Response:
            response = handler(request)
            if asyncio.iscoroutine(response):
                response = await response
            return response

        monkeypatch.setattr(
            checker.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(_handle), **kw),
        )

    return install


class TestDefiniteFailuresBlock:
    @pytest.mark.asyncio
    async def test_slow_judge_hits_deadline_and_blocks_creation(
        self,
        client,
        real_check,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(config_mod.settings, "model_check_deadline_seconds", 0.5)

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "judge":
                await asyncio.sleep(10)
            return httpx.Response(200, json=OPENAI_OK)

        real_check(handler)
        resp = await client.post("/api/v1/jobs/eval", json=EVAL_BODY)
        assert resp.status_code == 422
        body = resp.json()
        assert body["code"] == "model_check_failed"
        assert "judge model 'gpt-judge'" in body["message"] and "deadline" in body["message"]
        assert [d["field"] for d in body["details"]] == ["judge"]
        assert await _job_count() == 0

    @pytest.mark.asyncio
    async def test_validate_reports_the_same_failure(self, client, real_check) -> None:
        real_check(
            lambda r: (
                httpx.Response(404)
                if r.url.host == "judge"
                else httpx.Response(200, json=OPENAI_OK)
            )
        )
        resp = await client.post("/api/v1/jobs/eval/validate", json=EVAL_BODY)
        assert resp.status_code == 422
        assert resp.json()["code"] == "model_check_failed"
        assert "not_found" in resp.json()["message"]

    @pytest.mark.asyncio
    async def test_failure_message_never_carries_the_key(self, client, real_check) -> None:
        real_check(
            lambda r: (
                httpx.Response(401)
                if r.url.host == "judge"
                else httpx.Response(200, json=OPENAI_OK)
            )
        )
        body = {**EVAL_BODY, "judge": {**EVAL_BODY["judge"], "api_key": "sk-inline-secret"}}
        resp = await client.post("/api/v1/jobs/eval", json=body)
        assert resp.status_code == 422
        assert "auth_failed" in resp.text
        assert "sk-inline-secret" not in resp.text

    @pytest.mark.asyncio
    async def test_config_error_blocks_with_the_same_code(self, client, real_check) -> None:
        real_check(lambda _r: httpx.Response(200, json=OPENAI_OK))
        body = {**EVAL_BODY, "judge": {"base_url": "anthropic", "model": "claude"}}
        resp = await client.post("/api/v1/jobs/eval", json=body)
        assert resp.status_code == 422
        assert resp.json()["code"] == "model_check_failed"
        assert resp.json()["details"][0]["field"] == "judge"
        assert await _job_count() == 0


class TestMalformedUrls:
    @pytest.mark.asyncio
    async def test_malformed_port_is_model_check_failed_not_a_500(self, client, real_check) -> None:
        real_check(lambda _r: httpx.Response(200, json=OPENAI_OK))
        body = {**EVAL_BODY, "judge": {"base_url": "http://judge:invalid/v1", "model": "j"}}
        for path in ("/api/v1/jobs/eval/validate", "/api/v1/jobs/eval"):
            resp = await client.post(path, json=body)
            assert resp.status_code == 422, (path, resp.text)
            assert resp.json()["code"] == "model_check_failed"


class TestUnverifiedProceeds:
    UNVERIFIED_JUDGE = CheckResult(
        "judge",
        "gpt-judge",
        "http://judge:8000/v1",
        UNVERIFIED,
        "checker_unavailable",
        "model check service unreachable (ConnectError)",
    )

    @pytest.mark.asyncio
    async def test_created_job_records_and_returns_the_warning(self, client, monkeypatch) -> None:
        _stub_report(monkeypatch, CheckReport(results=[self.UNVERIFIED_JUDGE]))
        resp = await client.post("/api/v1/jobs/eval", json=EVAL_BODY)
        assert resp.status_code == 201
        checks = resp.json()["model_checks"]
        assert [(c["label"], c["outcome"], c["reason"]) for c in checks] == [
            ("judge", "unverified", "checker_unavailable")
        ]
        detail = await client.get(f"/api/v1/jobs/{resp.json()['id']}")
        assert detail.json()["model_checks"][0]["outcome"] == "unverified"  # survives a refresh

    @pytest.mark.asyncio
    async def test_validate_returns_the_warning_on_the_card(self, client, monkeypatch) -> None:
        _stub_report(monkeypatch, CheckReport(results=[self.UNVERIFIED_JUDGE]))
        resp = await client.post("/api/v1/jobs/eval/validate", json=EVAL_BODY)
        assert resp.status_code == 200
        assert resp.json()["model_checks"][0]["reason"] == "checker_unavailable"


class TestRetryIsAdmitted:
    async def _failed_eval(self, client, body: dict[str, Any] | None = None) -> str:
        resp = await client.post("/api/v1/jobs/eval", json=body or EVAL_BODY)
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]
        import amortized.db.connection as db

        async with db._pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET status = 'failed' WHERE id = $1", job_id)
        return job_id

    @pytest.mark.asyncio
    async def test_retry_is_model_checked(self, client, monkeypatch) -> None:
        job_id = await self._failed_eval(client)
        dead = CheckResult("judge", "gpt-judge", "http://judge:8000/v1", FAILED, "unreachable")
        seen = _stub_report(monkeypatch, CheckReport(results=[dead]))
        resp = await client.post(f"/api/v1/jobs/{job_id}/retry")
        assert resp.status_code == 422
        assert resp.json()["code"] == "model_check_failed"
        assert seen and seen[0]["job_type"] == "eval"


class TestConnectionScoping:
    @pytest.mark.asyncio
    async def test_no_pool_connection_is_held_while_checking(self, client, monkeypatch) -> None:
        from amortized.db.connection import get_pool

        in_use: list[int] = []

        async def fake(job_type: str, config: dict[str, Any]) -> CheckReport:
            pool = get_pool()
            in_use.append(pool.get_size() - pool.get_idle_size())
            return CheckReport()

        monkeypatch.setattr(job_admission, "check_job_models", fake)
        assert (await client.post("/api/v1/jobs/eval/validate", json=EVAL_BODY)).status_code == 200
        assert (await client.post("/api/v1/jobs/eval", json=EVAL_BODY)).status_code == 201
        assert in_use == [0, 0]

    @pytest.mark.asyncio
    async def test_prechecks_rerun_after_the_check(self, client, monkeypatch) -> None:
        import amortized.db.connection as db

        async with db._pool.acquire() as conn:
            from amortized.db.repository import Repository

            training = await create_job(
                Repository(conn), job_type=JobType.training, config={"algorithm": "sft"}
            )
            await conn.execute("UPDATE jobs SET status = 'succeeded' WHERE id = $1", training["id"])

        async def flip_upstream(job_type: str, config: dict[str, Any]) -> CheckReport:
            # The upstream job changes state while the (slow) check runs.
            async with db._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE jobs SET status = 'failed' WHERE id = $1", training["id"]
                )
            return CheckReport()

        monkeypatch.setattr(job_admission, "check_job_models", flip_upstream)
        body = {**EVAL_BODY, "training_job_id": training["id"]}
        resp = await client.post("/api/v1/jobs/eval", json=body)
        assert resp.status_code == 422
        assert await _job_count() == 1  # only the training job; the eval was not created


@pytest.mark.asyncio
async def test_create_job_refuses_hook_types_outside_admission() -> None:
    with pytest.raises(InvalidJobStateError, match="job admission"):
        await create_job(object(), job_type=JobType.eval, config={})  # type: ignore[arg-type]
