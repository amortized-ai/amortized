"""Morty's Eval Hub tools against the validation they lead to: a benchmark offered for a
model passes validation, and an unreachable Hub is a retryable error, never a 500."""

import copy
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import yaml
from fastapi import HTTPException

import amortized.config as config_mod
from amortized.api import evals
from amortized.api.jobs import _pin_benchmark
from amortized.integrations.evalhub import catalog, client

FILES = Path(__file__).resolve().parents[1] / "deploy/helm/amortized/files/evalhub"
LE = "amortized-lighteval"
ENDPOINT = {"base_url": "https://llm.example.com/v1", "model": "m"}
DECLARED = ["completions_logprobs"]


class FakeHub:
    """The documented REST shapes (catalog reads) and SDK provider reads."""

    def __init__(self) -> None:
        compat = json.loads((FILES / f"compat/{LE}.json").read_text())
        self.provider = {
            "resource": {"id": LE},
            "name": "Lighteval",
            "title": "Lighteval",
            "runtime": {"k8s": {"Image": compat["adapter_image"]}},
            "benchmarks": copy.deepcopy(
                yaml.safe_load((FILES / "lighteval.yaml").read_text())["benchmarks"]
            ),
        }
        self.down = False
        self.providers = SimpleNamespace(get=self._get_provider)

    async def _request_get(self, path: str, **_: object) -> MagicMock:
        if self.down:
            raise httpx.ConnectError("hub down")
        if path != f"/evaluations/providers/{LE}":
            raise httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
        response = MagicMock()
        response.json.return_value = self.provider
        return response

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
        evalhub_providers=[LE],
        evalhub_compat_dir=str(FILES / "compat"),
        evalhub_staging_bucket="staging",
    )
    monkeypatch.setattr(config_mod, "settings", settings)
    fake = FakeHub()

    @asynccontextmanager
    async def sdk_client(*_: object):
        yield fake

    monkeypatch.setattr(client, "client", sdk_client)
    return fake


def _benchmark(benchmark_id: str, **extra: object) -> dict[str, object]:
    return {"kind": "benchmark", "provider_id": LE, "benchmark_id": benchmark_id} | extra


# --------------------------------------------------------------------------- #
# Declared endpoint capabilities: what browsing offers, validation accepts
# --------------------------------------------------------------------------- #


async def test_every_benchmark_offered_for_a_declared_endpoint_passes_validation(hub):
    model = evals.BenchmarkModel(endpoint=ENDPOINT, declared_capabilities=DECLARED)
    declared = catalog.Capability.from_model(model.model_dump())
    chat_only = catalog.Capability.from_model({"endpoint": ENDPOINT})
    entries = await catalog.load(hub)
    offered = [e.id for e in entries if catalog.compatibility(e, declared)[0] == "compatible"]
    arc = next(e for e in entries if e.id == "arc:easy")
    assert "arc:easy" in offered  # likelihood: offered only because it was declared
    assert catalog.compatibility(arc, chat_only)[0] == "incompatible"
    for benchmark_id in offered:
        # Inside the evaluation, as get_benchmark and the workflow say, or beside the model.
        evaluation = _benchmark(benchmark_id, declared_capabilities=DECLARED)
        inside = {"endpoint": ENDPOINT, "evaluation": evaluation}
        beside = {
            "endpoint": ENDPOINT,
            "declared_capabilities": DECLARED,
            "evaluation": _benchmark(benchmark_id),
        }
        for config in (inside, beside):
            recipe, errors = await _pin_benchmark(config, approved=False)
            assert errors == [], (benchmark_id, config)
            assert recipe is not None and recipe["declared_capabilities"] == DECLARED


async def test_undeclared_endpoint_is_refused_a_likelihood_benchmark(hub):
    config = {"endpoint": ENDPOINT, "evaluation": _benchmark("arc:easy")}
    recipe, errors = await _pin_benchmark(config, approved=False)
    assert recipe is None
    assert "declare that this endpoint supports completions with logprobs" in errors[0]


async def test_get_benchmark_says_where_the_declared_capabilities_go(hub):
    declared = evals.BenchmarkModel(endpoint=ENDPOINT, declared_capabilities=DECLARED)
    result = await evals.get_benchmark(
        evals.GetBenchmarkRequest(provider_id=LE, id="arc:easy", model=declared)
    )
    assert result["compatibility"] == "compatible"
    assert 'changed, declared_capabilities: ["completions_logprobs"]}' in result["next"]
    served = evals.BenchmarkModel(training_job_id="job-1")
    result = await evals.get_benchmark(
        evals.GetBenchmarkRequest(provider_id=LE, id="arc:easy", model=served)
    )
    assert "declared_capabilities" not in result["next"]


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


async def test_hub_taking_no_new_jobs_is_reported_disabled_and_refuses_benchmarks(hub, monkeypatch):
    monkeypatch.setattr(config_mod.settings, "evalhub_accept_new_jobs", False)
    hub.down = True  # nothing may be read from Hub
    assert (await evals.list_evaluation_providers())["enabled"] is False
    model = evals.BenchmarkModel(endpoint=ENDPOINT)
    with pytest.raises(HTTPException) as found:
        await evals.find_benchmarks(evals.FindBenchmarksRequest(model=model, name="gsm8k"))
    with pytest.raises(HTTPException) as got:
        await evals.get_benchmark(
            evals.GetBenchmarkRequest(provider_id=LE, id="gsm8k", model=model)
        )
    for refused in (found.value, got.value):
        assert refused.status_code == 422
        assert refused.detail == "Benchmark evaluation is not enabled on this deployment"
    config = {"endpoint": ENDPOINT, "evaluation": _benchmark("gsm8k")}
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
    assert "benchmark" not in errors[0]
    assert caplog.records[-1].exc_info is not None


async def test_benchmark_validation_is_retryable_and_logged_when_hub_is_down(hub, caplog):
    hub.down = True
    config = {"training_job_id": "job-1", "evaluation": _benchmark("gsm8k")}
    with caplog.at_level(logging.WARNING):
        recipe, errors = await client.benchmark_recipe(config)
    assert recipe is None
    assert errors == [
        "Cannot verify the selected benchmark against the Eval Hub catalog; retry validation"
    ]
    assert "Could not read the Eval Hub catalog" in caplog.text
    assert caplog.records[-1].exc_info is not None
