"""Morty's Eval Hub tools: an unreachable Hub is a retryable error, never a 500."""

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import HTTPException

import amortized.config as config_mod
from amortized.api import evals
from amortized.integrations.evalhub import client

ENDPOINT = {"base_url": "https://llm.example.com/v1", "model": "m"}


class FakeHub:
    """SDK provider reads."""

    def __init__(self) -> None:
        self.down = False
        self.providers = SimpleNamespace(get=self._get_provider)

    async def _get_provider(self, name: str) -> object:
        if self.down:
            raise httpx.ConnectError("hub down")
        raise httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())


@pytest.fixture
def hub(monkeypatch):
    settings = config_mod.Settings(
        compute_backend="kubernetes",
        evalhub_url="http://hub",
        mlflow_tracking_uri="https://mlflow",
        mlflow_tracking_token_file="/var/run/secrets/mlflow/token",
        mlflow_workspace="amortized-jobs",
        evalhub_staging_bucket="staging",
    )
    monkeypatch.setattr(config_mod, "settings", settings)
    fake = FakeHub()

    @asynccontextmanager
    async def sdk_client(*_: object):
        yield fake

    monkeypatch.setattr(client, "client", sdk_client)
    return fake


# --------------------------------------------------------------------------- #
# An unreachable Hub: retryable, logged, and worded for what was being checked
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("down", [True, False], ids=["hub down", "provider missing"])
async def test_providers_are_a_retryable_503_when_hub_cannot_be_read(hub, caplog, down):
    hub.down = down
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException) as failure:
        await evals.list_evaluation_providers()
    assert failure.value.status_code == 503
    assert "Could not read the Eval Hub providers" in caplog.text
    assert caplog.records[-1].exc_info is not None


async def test_providers_report_hub_disabled_without_reading_it(monkeypatch):
    monkeypatch.setattr(config_mod, "settings", config_mod.Settings(evalhub_url=""))
    assert (await evals.list_evaluation_providers())["enabled"] is False


async def test_hub_taking_no_new_jobs_is_reported_disabled(hub, monkeypatch):
    monkeypatch.setattr(config_mod.settings, "evalhub_accept_new_jobs", False)
    hub.down = True  # nothing may be read from Hub
    assert (await evals.list_evaluation_providers())["enabled"] is False
    config = {"endpoint": ENDPOINT, "evaluation": {"kind": "task"}}
    assert await client.validate_definition(config) == [
        "evaluation definitions require the configured Eval Hub engine"
    ]


async def test_task_validation_names_the_task_provider_when_hub_is_down(hub, caplog):
    hub.down = True
    with caplog.at_level(logging.WARNING):
        errors = await client.validate_definition({"rubric": [{"name": "correct"}]})
    assert errors == [
        "Cannot verify the task evaluation against Eval Hub's provider 'amortized';"
        " retry validation"
    ]
    assert caplog.records[-1].exc_info is not None
