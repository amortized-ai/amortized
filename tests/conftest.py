"""Shared test fixtures and markers."""

import os

import pytest

import amortized.db.connection as db_conn_mod

TEST_DATABASE_URL = os.environ.get(
    "AMORTIZED_TEST_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized_test",
)


@pytest.fixture(autouse=True)
async def _reset_db() -> None:
    """Close the connection pool between tests so each test gets a fresh one."""
    yield
    if db_conn_mod._pool is not None:
        await db_conn_mod._pool.close()
        db_conn_mod._pool = None


@pytest.fixture(autouse=True)
def _offline_model_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Job admission checks a job's LLM endpoints. Keep the suite offline by reporting
    nothing to check; tests of the check itself patch around this."""
    from amortized.core.model_check_client import CheckReport

    async def _nothing_to_check(_job_type: str, _config: dict) -> CheckReport:
        return CheckReport()

    monkeypatch.setattr("amortized.api.job_admission.check_job_models", _nothing_to_check)
