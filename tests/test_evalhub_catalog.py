"""Benchmark catalog: the vendored Eval Hub catalog, the helper's compatibility file,
browsing, exact names, recipe pinning and the two Morty tools."""

import copy
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yaml

import amortized.config as config_mod
from amortized.integrations.evalhub import catalog, client
from amortized.jobs.base import JobBuildError

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "deploy/helm/amortized/files/evalhub"
CATALOG = yaml.safe_load((FILES / "lighteval.yaml").read_text())
COMPAT = json.loads((FILES / "compat/amortized-lighteval.json").read_text())
IMAGE = COMPAT["adapter_image"]
SERVED = catalog.Capability.from_model({"training_job_id": "job-1"})
ENDPOINT = catalog.Capability.from_model({"endpoint": {"base_url": "http://m/v1", "model": "m"}})
SHORTLIST = {
    "resource": {"id": "amortized-benchmarks-v1"},
    "name": "Amortized benchmarks v1",
    "benchmarks": [{"provider_id": "amortized-lighteval", "id": "gsm8k"}],
}


class FakeHub:
    """Serves the documented REST shapes the catalog reads (raw, not the SDK models)."""

    def __init__(self, image: str = IMAGE) -> None:
        self.provider = {
            "resource": {"id": "amortized-lighteval"},
            "name": "Lighteval",
            "title": "Lighteval",
            "runtime": {"k8s": {"Image": image}},
            "benchmarks": copy.deepcopy(CATALOG["benchmarks"]),
        }
        self.fail = False

    async def _request_get(self, path: str, **_: object) -> MagicMock:
        if self.fail:
            raise httpx.ConnectError("hub down")
        response = MagicMock()
        if path == "/evaluations/collections":
            response.json.return_value = {"items": [SHORTLIST]}
        elif path == "/evaluations/providers/amortized-lighteval":
            response.json.return_value = self.provider
        else:
            raise httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
        return response


@pytest.fixture
def hub(monkeypatch):
    settings = config_mod.Settings(
        compute_backend="kubernetes",
        evalhub_url="http://hub",
        mlflow_tracking_uri="http://mlflow",
        evalhub_providers=["amortized-lighteval"],
        evalhub_compat_dir=str(FILES / "compat"),
    )
    monkeypatch.setattr(config_mod, "settings", settings)
    fake = FakeHub()

    @asynccontextmanager
    async def sdk_client(*_: object):
        yield fake

    monkeypatch.setattr(client, "client", sdk_client)
    return fake


async def entries(fake: FakeHub) -> list[catalog.Entry]:
    return await catalog.load(fake)


# --------------------------------------------------------------------------- #
# Release artifacts stay consistent with each other and with the chart
# --------------------------------------------------------------------------- #


def test_compat_file_matches_the_pinned_image_and_catalog():
    values = (ROOT / "deploy/helm/amortized/values.yaml").read_text()
    pinned = re.search(r"lightEvalImage:\s*(\S+)", values).group(1)
    assert COMPAT["schema"] == catalog.SCHEMA
    assert COMPAT["adapter_image"] == pinned
    assert COMPAT["adapter_digest"] == catalog.image_digest(pinned)
    assert set(COMPAT["benchmarks"]) == {b["id"] for b in CATALOG["benchmarks"]}
    commit = COMPAT["catalog"]["commit"]
    assert commit in (FILES / "lighteval.yaml").read_text().splitlines()[0]
    assert f"EVALHUB_CATALOG_COMMIT ?= {commit}" in (ROOT / "Makefile").read_text()
    for record in COMPAT["benchmarks"].values():
        if record["status"] == "known":
            assert record["settings_profile"] in catalog.SETTINGS_PROFILES
            assert record["scoring"] in {"generative", "likelihood"}


# --------------------------------------------------------------------------- #
# Compatibility: hard filters before anything is shown
# --------------------------------------------------------------------------- #


async def test_served_model_hides_only_suites(hub):
    result = catalog.find(await entries(hub), SERVED)
    assert result["hidden"] == [{"reason": "suites are not supported yet", "count": 7}]
    assert result["available"] == 21


async def test_endpoint_hides_likelihood_benchmarks_unless_declared(hub):
    loaded = await entries(hub)
    result = catalog.find(loaded, ENDPOINT)
    likelihood = sum(
        1
        for r in COMPAT["benchmarks"].values()
        if r.get("scoring") == "likelihood" and not r.get("suite")
    )
    assert likelihood == 7
    assert {h["count"] for h in result["hidden"]} == {7}
    assert result["available"] == 14
    declared = catalog.Capability.from_model(
        {"endpoint": {"base_url": "http://m/v1"}, "declared_capabilities": ["completions_logprobs"]}
    )
    assert catalog.find(loaded, declared)["available"] == 21


def test_model_must_be_settled_first():
    with pytest.raises(ValueError, match="Settle the model first"):
        catalog.Capability.from_model({})


async def test_digest_mismatch_makes_every_entry_unknown(hub):
    hub.provider["runtime"]["k8s"]["Image"] = "quay.io/x/lighteval@sha256:other"
    loaded = await entries(hub)
    gsm8k = next(e for e in loaded if e.id == "gsm8k")
    status, reason = catalog.compatibility(gsm8k, SERVED)
    assert status == "unknown"
    assert "make evalhub-compat" in reason


# --------------------------------------------------------------------------- #
# Exact names: never a substitute
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "capability", "expected"),
    [
        ("GSM-8K", ENDPOINT, ("gsm8k", "compatible")),
        ("lighteval arc challenge", ENDPOINT, ("arc:challenge", "incompatible")),
        ("lighteval arc challenge", SERVED, ("arc:challenge", "compatible")),
        ("mmlu", SERVED, ("mmlu", "unknown")),
        ("math", SERVED, ("math", "unsupported")),
    ],
)
async def test_name_lookup_is_exact(hub, name, capability, expected):
    result = catalog.find(await entries(hub), capability, name=name)
    assert result["step"] == "matches"
    assert [(b["id"], b["compatibility"]) for b in result["benchmarks"]] == [expected]


async def test_unknown_name_offers_no_substitute(hub):
    result = catalog.find(await entries(hub), SERVED, name="gsm9k")
    assert result["benchmarks"] == []
    assert "do not suggest a substitute" in result["next"]


# --------------------------------------------------------------------------- #
# Browsing: area -> kind -> provider; card when <= 4, typed question otherwise
# --------------------------------------------------------------------------- #


async def test_browse_drills_down_to_a_card_then_selects(hub):
    loaded = await entries(hub)
    shortlists = await catalog.load_shortlists(hub)
    first = catalog.find(loaded, SERVED, shortlists=shortlists)
    assert (first["step"], first["facet"], first["display"]) == ("question", "area", "card")
    titles = [o["title"] for o in first["options"]]
    assert titles[0] == "Knowledge & reasoning"
    assert "Shortlist: Amortized benchmarks v1" in titles
    assert len(first["options"]) <= catalog.CARD_LIMIT

    kind = catalog.find(
        loaded, SERVED, picks=first["picks"], answer="Knowledge & reasoning", shortlists=shortlists
    )
    assert (kind["facet"], kind["picks"]) == ("kind", {"area": "knowledge_and_reasoning"})
    # Shortlists are offered only at the first question.
    assert not any(o["title"].startswith("Shortlist") for o in kind["options"])

    math = catalog.find(loaded, SERVED, picks=kind["picks"], answer="Math", shortlists=shortlists)
    assert (math["step"], math["display"]) == ("benchmarks", "card")
    assert "gsm8k" in [o["value"] for o in math["options"]]

    selected = catalog.find(loaded, SERVED, picks=math["picks"], answer="gsm8k")
    assert selected["step"] == "selected"
    assert selected["benchmark"]["id"] == "gsm8k"


async def test_more_than_four_benchmarks_become_a_typed_list(hub):
    loaded = await entries(hub)
    picks = {"area": "knowledge_and_reasoning"}
    reasoning = catalog.find(loaded, SERVED, picks=picks, answer="Reasoning")
    assert (reasoning["step"], reasoning["display"]) == ("benchmarks", "text")
    assert len(reasoning["benchmarks"]) > catalog.CARD_LIMIT
    assert "options" not in reasoning
    chosen = catalog.find(loaded, SERVED, picks=reasoning["picks"], answer="ARC challenge")
    assert chosen["benchmark"]["id"] == "arc:challenge"


def test_more_than_four_choices_become_a_typed_question():
    provider = catalog.Provider("p", "P", "img@sha256:x", {}, "")
    known = {
        "status": "known",
        "scoring": "generative",
        "settings_profile": "lighteval-endpoint-v1",
    }

    def entry(i: int, kind: str) -> catalog.Entry:
        raw = {
            "id": f"b{i}",
            "category": kind,
            "domains": ["knowledge_and_reasoning"],
            "evaluation_targets": ["model"],
            "modalities": ["text"],
        }
        return catalog.Entry(provider, raw, known)

    pool = [entry(i, kind) for i, kind in enumerate("abcdef" * 2)]
    result = catalog.find(pool, SERVED)
    assert (result["step"], result["facet"], result["display"]) == ("question", "kind", "text")
    assert [c["count"] for c in result["choices"]] == [2] * 6
    assert "Do not use present_options" in result["next"]


async def test_off_list_answer_is_rejected_with_the_allowed_values(hub):
    loaded = await entries(hub)
    with pytest.raises(catalog.ChoiceError, match="Knowledge & reasoning"):
        catalog.find(loaded, SERVED, answer="Astrology")
    with pytest.raises(catalog.ChoiceError, match="not a area"):
        catalog.find(loaded, SERVED, picks={"area": "astrology"})


async def test_shortlist_narrows_to_its_members(hub):
    loaded = await entries(hub)
    shortlists = await catalog.load_shortlists(hub)
    result = catalog.find(
        loaded, SERVED, answer="Shortlist: Amortized benchmarks v1", shortlists=shortlists
    )
    assert result["picks"] == {"shortlist": "amortized-benchmarks-v1"}
    assert [o["value"] for o in result["options"]] == ["gsm8k"]


# --------------------------------------------------------------------------- #
# Recipe pinning
# --------------------------------------------------------------------------- #


def definition(benchmark_id: str = "gsm8k", **parameters: object) -> dict:
    return {
        "kind": "benchmark",
        "provider_id": "amortized-lighteval",
        "benchmark_id": benchmark_id,
        "parameters": parameters,
    }


async def test_recipe_sends_the_catalog_few_shot_and_no_per_question_results(hub):
    loaded = await entries(hub)
    recipe, errors = catalog.resolve_recipe(loaded, ENDPOINT, definition(), {"max_samples": 200})
    assert errors == []
    assert recipe["parameters"] == {
        "provider": "endpoint",
        "max_samples": 200,
        "num_few_shot": 8,
        "batch_size": 1,
        "save_sample_results": False,
    }
    assert recipe["adapter_digest"] == COMPAT["adapter_digest"]
    assert recipe["catalog_entry"]["primary_score"]["metric"] == "gsm8k.extractive_match"


async def test_fingerprint_ignores_serving_but_not_settings(hub):
    loaded = await entries(hub)
    served, _ = catalog.resolve_recipe(loaded, SERVED, definition(), {})
    endpoint, _ = catalog.resolve_recipe(loaded, ENDPOINT, definition(), {})
    zero_shot, _ = catalog.resolve_recipe(loaded, ENDPOINT, definition(num_few_shot=0), {})
    assert catalog.fingerprint(served) == catalog.fingerprint(endpoint)
    assert catalog.fingerprint(zero_shot) != catalog.fingerprint(endpoint)


@pytest.mark.parametrize(
    ("benchmark_id", "parameters", "message"),
    [
        ("arc:challenge", {}, "incompatible with this model"),
        ("mmlu", {}, "unknown compatibility"),
        ("math", {}, "suites are not supported yet"),
        ("nope", {}, "is not in provider"),
        ("gsm8k", {"num_few_shot": 99}, "num_few_shot"),
    ],
)
async def test_recipe_refuses_what_cannot_be_approved(hub, benchmark_id, parameters, message):
    recipe, errors = catalog.resolve_recipe(
        await entries(hub), ENDPOINT, definition(benchmark_id, **parameters), {}
    )
    assert recipe is None
    assert message in errors[0]


async def test_benchmark_recipe_reports_hub_failures(hub):
    hub.fail = True
    recipe, errors = await client.benchmark_recipe(
        {"endpoint": {"base_url": "http://m/v1", "model": "m"}, "evaluation": definition()}
    )
    assert recipe is None
    assert "retry validation" in errors[0]


async def test_validate_definition_leaves_catalog_checks_to_the_recipe(hub):
    config = {"endpoint": {"base_url": "http://m.example/v1", "model": "m"}}
    assert await client.validate_definition({**config, "evaluation": definition()}) == []
    other = {**definition(), "provider_id": "lighteval"}
    assert "not enabled" in (await client.validate_definition({**config, "evaluation": other}))[0]


# --------------------------------------------------------------------------- #
# Before sending to Hub: the approved recipe, and the approved adapter
# --------------------------------------------------------------------------- #


async def test_hub_job_is_built_only_from_an_unchanged_approved_recipe(hub):
    from amortized.integrations.evalhub.preparation import _approved_recipe

    recipe, _ = catalog.resolve_recipe(await entries(hub), ENDPOINT, definition(), {})
    provider = {"runtime": {"k8s": {"Image": IMAGE}}}
    assert _approved_recipe({**definition(), "recipe": recipe}, provider) is recipe
    with pytest.raises(JobBuildError, match="changed after approval"):
        _approved_recipe(
            {**definition(), "recipe": recipe},
            {"runtime": {"k8s": {"image": "quay.io/x@sha256:other"}}},
        )
    with pytest.raises(JobBuildError, match="no approved recipe"):
        _approved_recipe(definition(), provider)


async def test_benchmark_comparisons_key_on_the_recipe_fingerprint(monkeypatch):
    from test_evalhub import snapshot

    from amortized.integrations.evalhub import results

    def job(job_id: str, model: dict, fingerprint: str) -> dict:
        evaluation = {**definition(), "recipe_fingerprint": fingerprint}
        return {
            "id": job_id,
            "config": {**model, "evaluation": evaluation},
            "status": "succeeded",
            "created_at": "now",
        }

    execution = {
        "snapshot": snapshot("completed", metrics={"gsm8k.extractive_match": 0.5}),
        "request": {},
        "inputs": {"catalog": {"runtime": {"image": IMAGE}}},
    }
    executions = {k: copy.deepcopy(execution) for k in "abc"}
    executions["b"]["inputs"]["catalog"]["description"] = "unrelated catalog edit"
    store = MagicMock()
    store.get = AsyncMock(side_effect=lambda job_id: executions[job_id])
    monkeypatch.setattr(results, "EvalHubRepository", lambda _: store)
    base = job("a", {"endpoint": {"base_url": "http://m/v1", "model": "base"}}, "fp1")
    tuned = job("b", {"training_job_id": "t1"}, "fp1")
    groups = await results.comparison_groups(MagicMock(), [base, tuned])
    assert len(groups) == 1 and len(groups[0]["evals"]) == 2
    changed = job("c", {"training_job_id": "t1"}, "fp2")
    assert len(await results.comparison_groups(MagicMock(), [base, changed])) == 2


# --------------------------------------------------------------------------- #
# The two Morty tools
# --------------------------------------------------------------------------- #


@pytest.fixture
async def api(hub):
    from fastapi import FastAPI

    from amortized.api.evals import router

    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as transport:
        yield transport


MODEL = {"endpoint": {"base_url": "http://m/v1", "model": "m"}}


async def test_find_benchmarks_tool(api, hub):
    first = await api.post("/api/v1/evaluations/benchmarks/find", json={"model": MODEL})
    assert first.status_code == 200, first.text
    assert first.json()["step"] == "question"
    bad = await api.post(
        "/api/v1/evaluations/benchmarks/find",
        json={"model": MODEL, "picks": first.json()["picks"], "answer": "Astrology"},
    )
    assert bad.status_code == 422
    assert "Knowledge & reasoning" in bad.json()["detail"]
    unsettled = await api.post("/api/v1/evaluations/benchmarks/find", json={"model": {}})
    assert unsettled.status_code == 422
    assert "Settle the model first" in unsettled.json()["detail"]
    hub.fail = True
    down = await api.post("/api/v1/evaluations/benchmarks/find", json={"model": MODEL})
    assert down.status_code == 503


async def test_get_benchmark_tool(api):
    url = "/api/v1/evaluations/benchmarks/get"
    gsm8k = await api.post(
        url, json={"provider_id": "amortized-lighteval", "id": "gsm8k", "model": MODEL}
    )
    assert gsm8k.status_code == 200, gsm8k.text
    body = gsm8k.json()
    assert body["compatibility"] == "compatible"
    settings = {s["name"]: s for s in body["settings"]}
    assert settings["num_few_shot"]["default"] == 8
    assert settings["num_few_shot"]["max"] == 50
    assert settings["max_samples"]["default"] == "all questions"
    assert "validate_eval_job" in body["next"]
    arc = await api.post(
        url, json={"provider_id": "amortized-lighteval", "id": "arc:challenge", "model": MODEL}
    )
    assert arc.json()["compatibility"] == "incompatible"
    assert "do not offer a different benchmark" in arc.json()["next"]
    missing = await api.post(
        url, json={"provider_id": "amortized-lighteval", "id": "nope", "model": MODEL}
    )
    assert missing.status_code == 404
    other = await api.post(url, json={"provider_id": "lighteval", "id": "gsm8k", "model": MODEL})
    assert other.status_code == 404
