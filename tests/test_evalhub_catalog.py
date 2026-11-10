"""Benchmark catalog: the vendored Eval Hub catalogs, the helpers' compatibility files,
browsing, exact names, copies across providers, recipe pinning and the two Morty tools."""

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
from amortized.integrations.evalhub import SERVER_LIMIT, catalog, client
from amortized.jobs.base import JobBuildError

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "deploy/helm/amortized/files/evalhub"
CATALOG = yaml.safe_load((FILES / "lighteval.yaml").read_text())
COMPAT = json.loads((FILES / "compat/amortized-lighteval.json").read_text())
IMAGE = COMPAT["adapter_image"]
LM_CATALOG = yaml.safe_load((FILES / "lm_evaluation_harness.yaml").read_text())
LM_COMPAT = json.loads((FILES / "compat/amortized-lm-eval.json").read_text())
# Release artifacts per registered provider: (vendored catalog, values.yaml image key).
ARTIFACTS = {
    "amortized-lighteval": ("lighteval.yaml", "lightEvalImage"),
    "amortized-lm-eval": ("lm_evaluation_harness.yaml", "lmEvalImage"),
}
SERVED = catalog.Capability.from_model({"training_job_id": "job-1"})
ENDPOINT = catalog.Capability.from_model({"endpoint": {"base_url": "http://m/v1", "model": "m"}})


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
        self.lm_eval = {
            "resource": {"id": "amortized-lm-eval"},
            "name": "LM Evaluation Harness",
            "title": "LM Evaluation Harness",
            "runtime": {"k8s": {"Image": LM_COMPAT["adapter_image"]}},
            "benchmarks": copy.deepcopy(LM_CATALOG["benchmarks"]),
        }
        self.fail = False

    async def _request_get(self, path: str, **_: object) -> MagicMock:
        if self.fail:
            raise httpx.ConnectError("hub down")
        response = MagicMock()
        if path == "/evaluations/providers/amortized-lighteval":
            response.json.return_value = self.provider
        elif path == "/evaluations/providers/amortized-lm-eval":
            response.json.return_value = self.lm_eval
        else:
            raise httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
        return response


@pytest.fixture
def hub(monkeypatch):
    settings = config_mod.Settings(
        compute_backend="kubernetes",
        evalhub_url="http://hub",
        mlflow_tracking_uri="https://mlflow",
        mlflow_tracking_token_file="/var/run/secrets/mlflow/token",
        mlflow_workspace="amortized-jobs",
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


@pytest.fixture
def both(hub):
    """lm-eval first, as deployed: it is the default copy of a shared benchmark."""
    config_mod.settings.evalhub_providers = ["amortized-lm-eval", "amortized-lighteval"]
    return hub


def _contract(tmp_path: Path, change) -> str:
    """A copy of the chart's compat files, changed by `change(lm_eval, lighteval)`."""
    lm, le = copy.deepcopy(LM_COMPAT), copy.deepcopy(COMPAT)
    change(lm["benchmarks"], le["benchmarks"])
    (tmp_path / "amortized-lm-eval.json").write_text(json.dumps(lm))
    (tmp_path / "amortized-lighteval.json").write_text(json.dumps(le))
    return str(tmp_path)


@pytest.fixture
def copies(both, tmp_path):
    """The pinned images have no copies: ARC Easy is scored differently and lm-eval cannot
    load truthful_qa. To test how copies are offered, ARC Easy's two entries are confirmed
    as scoring alike."""

    def confirm(lm, le):
        lm["arc_easy"]["same_scoring"] = [{"provider_id": "amortized-lighteval", "id": "arc:easy"}]
        le["arc:easy"]["same_scoring"] = [{"provider_id": "amortized-lm-eval", "id": "arc_easy"}]

    config_mod.settings.evalhub_compat_dir = _contract(tmp_path, confirm)
    return both


# --------------------------------------------------------------------------- #
# Release artifacts stay consistent with each other and with the chart
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider_id", sorted(ARTIFACTS))
def test_compat_file_matches_the_pinned_image_and_catalog(provider_id):
    catalog_file, image_key = ARTIFACTS[provider_id]
    compat = json.loads((FILES / f"compat/{provider_id}.json").read_text())
    raw = yaml.safe_load((FILES / catalog_file).read_text())
    values = (ROOT / "deploy/helm/amortized/values.yaml").read_text()
    pinned = re.search(rf"{image_key}:\s*(\S+)", values).group(1)
    assert compat["schema"] == catalog.SCHEMA
    assert compat["adapter_image"] == pinned
    assert compat["adapter_digest"] == catalog.image_digest(pinned)
    assert compat["settings_profile"] in catalog.SETTINGS_PROFILES
    assert set(compat["benchmarks"]) == {b["id"] for b in raw["benchmarks"]}
    commit = compat["catalog"]["commit"]
    assert commit in (FILES / catalog_file).read_text().splitlines()[0]
    assert f"EVALHUB_CATALOG_COMMIT ?= {commit}" in (ROOT / "Makefile").read_text()
    for record in compat["benchmarks"].values():
        if record["status"] == "known" and not record.get("suite"):
            assert record["settings_profile"] == compat["settings_profile"]
            assert record["scoring"] in {"generative", "likelihood"}
            for dataset in record["datasets"]:
                assert dataset["hf_repo"] and dataset["splits"], dataset


@pytest.mark.parametrize(
    ("records", "benchmark_id", "metric"),
    [
        (LM_COMPAT, "arc_easy", "acc_norm"),  # the catalog's name, reported as is
        (COMPAT, "gsm8k", "gsm8k.extractive_match"),
        (COMPAT, "glue:sst2", "glue:sst2.acc"),  # LightEval prefixes the task
        (COMPAT, "glue:cola", "glue:cola.mcc"),  # matthews_correlation under its alias
        (COMPAT, "arc:easy", "arc:easy.acc"),  # names acc_norm; acc is all it reports
        (COMPAT, "truthfulqa:mc", "truthfulqa:mc.truthfulqa_mc1"),  # mc1 under its alias
        (COMPAT, "math:algebra", "math:algebra.maj@4"),  # the only metric it reports
    ],
)
def test_score_metric_follows_what_the_pinned_image_reports(records, benchmark_id, metric):
    assert records["benchmarks"][benchmark_id]["score"]["metric"] == metric


def test_entry_score_comes_only_from_the_contract():
    raw = {"id": "b", "primary_score": {"metric": "acc_norm", "lower_is_better": False}}
    chosen = {"metric": "b.acc", "lower_is_better": False, "source": "only metric reported"}
    assert catalog.score({"score": chosen}) == {"metric": "b.acc", "lower_is_better": False}
    assert catalog.score({"score": {"metric": None, "reason": "r"}}) is None
    assert catalog.no_score_reason({"score": {"metric": None, "reason": "r"}}) == "r"
    # Never upstream's catalog: an entry the contract has not scored has no score.
    entry = catalog.Entry(catalog.Provider("p", "p", "", {}, ""), raw, {"status": "known"})
    assert catalog.score({}) is None and catalog.score(None) is None
    assert entry.metric == ""


async def test_an_entry_without_a_score_says_why(both, tmp_path):
    # No entry on the pinned images ran without a score (pile_* cannot load its dataset);
    # pile_arxiv is given the result it would have: three perplexities, none of them acc.
    reason = (
        "the catalog's metric is acc; the adapter reports bits_per_byte, byte_perplexity,"
        " word_perplexity"
    )

    def no_score(lm, le):
        lm["pile_arxiv"] = {
            **{k: v for k, v in lm["pile_arxiv"].items() if k != "reason"},
            "status": "known",
            "score": {"metric": None, "reason": reason},
        }

    config_mod.settings.evalhub_compat_dir = _contract(tmp_path, no_score)
    pile = next(e for e in await entries(both) if e.id == "pile_arxiv")
    shown = catalog.view(pile, SERVED)
    assert shown["metric"] == ""
    assert shown["no_score"] == reason


def test_enabled_providers_default_to_lm_eval_first():
    assert config_mod.Settings().evalhub_providers == ["amortized-lm-eval", "amortized-lighteval"]
    values = yaml.safe_load((ROOT / "deploy/helm/amortized/values.yaml").read_text())
    assert values["evalhub"]["providers"] == ["amortized-lm-eval", "amortized-lighteval"]


# --------------------------------------------------------------------------- #
# Compatibility: hard filters before anything is shown
# --------------------------------------------------------------------------- #


async def test_served_model_is_offered_only_verified_entries(hub):
    result = catalog.find(await entries(hub), SERVED)
    unverified = [
        k for k, r in COMPAT["benchmarks"].items() if not r.get("suite") and "verified" not in r
    ]
    assert sorted(unverified) == ["gpqa:diamond", "mmlu"]  # a gated dataset; no such task
    assert result["available"] == 19
    # Neither the unverified entries nor the suites are counted for Morty to repeat.
    assert "hidden" not in result


async def test_endpoint_hides_likelihood_benchmarks_unless_declared(hub):
    loaded = await entries(hub)
    result = catalog.find(loaded, ENDPOINT)
    likelihood = sum(
        1
        for r in COMPAT["benchmarks"].values()
        if r.get("scoring") == "likelihood" and not r.get("suite")
    )
    assert likelihood == 7
    assert result["available"] == 12
    assert "hidden" not in result  # something can run, so nothing to explain
    declared = catalog.Capability.from_model(
        {"endpoint": {"base_url": "http://m/v1"}, "declared_capabilities": ["completions_logprobs"]}
    )
    assert catalog.find(loaded, declared)["available"] == 19


def test_model_must_be_settled_first():
    with pytest.raises(ValueError, match="Settle the model first"):
        catalog.Capability.from_model({})


async def test_digest_mismatch_makes_every_entry_unknown(hub):
    hub.provider["runtime"]["k8s"]["Image"] = "quay.io/x/lighteval@sha256:other"
    loaded = await entries(hub)
    gsm8k = next(e for e in loaded if e.id == "gsm8k")
    status, reason = catalog.compatibility(gsm8k, SERVED)
    assert status == "unknown"
    assert "make evalhub-describe evalhub-conform evalhub-contract" in reason


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
# Browsing: area -> kind, never the framework; card when <= 4, typed question otherwise
# --------------------------------------------------------------------------- #


async def test_browse_drills_down_to_a_card_then_selects(hub):
    loaded = await entries(hub)
    first = catalog.find(loaded, SERVED)
    assert (first["step"], first["facet"], first["display"]) == ("question", "area", "card")
    assert first["options"][0]["title"] == "Knowledge & reasoning"
    assert len(first["options"]) <= catalog.CARD_LIMIT

    kind = catalog.find(loaded, SERVED, picks=first["picks"], answer="Knowledge & reasoning")
    assert (kind["facet"], kind["picks"]) == ("kind", {"area": "knowledge_and_reasoning"})

    math = catalog.find(loaded, SERVED, picks=kind["picks"], answer="Math")
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


async def test_browsing_never_asks_for_the_framework(both):
    loaded = await entries(both)

    def walk(result, path):
        if result["step"] != "question":
            assert result["step"] == "benchmarks", path
            return [result]
        assert result["facet"] in ("area", "kind"), path
        found = []
        for choice in result.get("options") or result.get("choices"):
            label = choice.get("title") or choice.get("label")
            found += walk(
                catalog.find(loaded, SERVED, picks=result["picks"], answer=label), [*path, label]
            )
        return found

    lists = walk(catalog.find(loaded, SERVED), [])
    assert sum(len(r["benchmarks"]) for r in lists) == catalog.find(loaded, SERVED)["available"]


async def test_two_frameworks_benchmarks_in_one_list_are_told_apart_by_id(both):
    loaded = await entries(both)
    reasoning = catalog.find(
        loaded, SERVED, picks={"area": "knowledge_and_reasoning"}, answer="Reasoning"
    )
    listed = {(b["provider_id"], b["id"]) for b in reasoning["benchmarks"]}
    assert {("amortized-lm-eval", "arc_easy"), ("amortized-lighteval", "arc:easy")} <= listed
    for answer, provider_id in (
        ("arc_easy", "amortized-lm-eval"),
        ("arc:easy", "amortized-lighteval"),
    ):
        chosen = catalog.find(loaded, SERVED, picks=reasoning["picks"], answer=answer)
        assert (chosen["benchmark"]["provider_id"], chosen["benchmark"]["id"]) == (
            provider_id,
            answer,
        )
    # Same name, both runnable: Morty asks which one by id and score, never by framework.
    with pytest.raises(catalog.ChoiceError) as ask:
        catalog.find(loaded, SERVED, picks=reasoning["picks"], answer="Basic science Q&A")
    assert "arc_easy (scored by acc_norm); arc:easy (scored by arc:easy.acc)" in str(ask.value)
    assert "framework" not in str(ask.value).lower()


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


async def test_question_counts_are_the_contracts_never_the_catalogs(both):
    loaded = await entries(both)
    lm_eval = catalog.find(loaded, SERVED, name="AraDiCE_ArabicMMLU_high_humanities_history_lev")
    # lm-eval's catalog says 14042 for every ArabicMMLU subject; its run reported 760.
    assert lm_eval["benchmarks"][0]["questions"] == 760
    # LightEval's catalog has no size for it; LightEval's loader counted 175.
    lcb = catalog.find(loaded, SERVED, name="lcb:codegeneration_v6")
    assert lcb["benchmarks"][0]["questions"] == 175
    # LightEval openbookqa asks its test and validation splits; the catalog says 500.
    assert catalog.find(loaded, SERVED, name="openbookqa")["benchmarks"][0]["questions"] == 1000


# --------------------------------------------------------------------------- #
# Copies of one benchmark in two providers: one entry, the first that can run
# --------------------------------------------------------------------------- #


def _ids(benchmarks: list[dict]) -> list[tuple[str, str, str]]:
    return [(b["provider_id"], b["id"], b["compatibility"]) for b in benchmarks]


async def test_same_questions_are_same_dataset_subset_and_split(both):
    loaded = await entries(both)
    groups = catalog.same_questions(loaded)
    # TruthfulQA would pair too, but lm-eval's truthfulqa_mc1 cannot load its dataset on
    # the pinned image, so it is unknown. ARC Easy pairs, but is scored differently.
    assert {tuple((e.provider.id, e.id) for e in g) for g in groups} == {
        (("amortized-lm-eval", "arc_easy"), ("amortized-lighteval", "arc:easy")),
    }
    assert catalog.overlaps(loaded) == {}


async def test_arc_easy_is_two_choices_each_with_its_own_score(both):
    result = catalog.find(await entries(both), SERVED, name="arc_easy")
    shown = {(b["provider_id"], b["id"]): b for b in result["benchmarks"]}
    assert set(shown) == {("amortized-lm-eval", "arc_easy"), ("amortized-lighteval", "arc:easy")}
    assert shown[("amortized-lm-eval", "arc_easy")]["metric"] == "acc_norm"
    assert shown[("amortized-lighteval", "arc:easy")]["metric"] == "arc:easy.acc"
    assert not any(b.get("also_in") for b in shown.values())


def test_a_provider_with_two_entries_on_one_dataset_has_no_copies():
    def entry(provider: str, benchmark_id: str) -> catalog.Entry:
        record = {
            "status": "known",
            "datasets": [{"hf_repo": "o/d", "hf_subset": "s", "splits": ["test"]}],
        }
        return catalog.Entry(
            catalog.Provider(provider, provider, "", {}, ""), {"id": benchmark_id}, record
        )

    def pairs(entries: list[catalog.Entry]) -> list[list[str]]:
        return [[f"{e.provider.id}/{e.id}" for e in g] for g in catalog.same_questions(entries)]

    assert pairs([entry("a", "x"), entry("b", "x")]) == [["a/x", "b/x"]]
    # mc1 and mc2 read the same questions but score them differently.
    assert pairs([entry("a", "mc1"), entry("a", "mc2"), entry("b", "mc")]) == []
    other_split = entry("b", "x")
    other_split.compat["datasets"][0]["splits"] = ["validation"]
    assert pairs([entry("a", "x"), other_split]) == []
    # The same questions make a copy only once the contract confirms the same scoring.
    a, b = entry("a", "x"), entry("b", "x")
    assert not catalog.overlaps([a, b])
    a.compat["same_scoring"] = [{"provider_id": "b", "id": "x"}]
    b.compat["same_scoring"] = [{"provider_id": "a", "id": "x"}]
    assert catalog.overlaps([a, b])


async def test_served_model_gets_the_lm_eval_copy(copies):
    loaded = await entries(copies)
    for name in ("arc_easy", "ARC:Easy"):
        result = catalog.find(loaded, SERVED, name=name)
        assert _ids(result["benchmarks"]) == [("amortized-lm-eval", "arc_easy", "compatible")]
        also = [{"provider_id": "amortized-lighteval", "id": "arc:easy"}]
        assert result["benchmarks"][0]["also_in"] == also
    browse = catalog.find(loaded, SERVED, picks={"area": "knowledge_and_reasoning"})
    area = [
        e
        for e in loaded
        if "knowledge_and_reasoning" in e.areas
        and catalog.compatibility(e, SERVED)[0] == "compatible"
    ]
    groups = catalog.overlaps(loaded)
    folded = [
        e
        for e in area
        if e.provider.id == "amortized-lighteval" and (e.provider.id, e.id) in groups
    ]
    assert [e.id for e in folded] == ["arc:easy"]
    assert browse["available"] == len(area) - len(folded)


async def test_naming_the_framework_picks_its_copy(copies):
    result = catalog.find(await entries(copies), SERVED, name="lighteval arc easy")
    assert _ids(result["benchmarks"]) == [("amortized-lighteval", "arc:easy", "compatible")]
    assert result["benchmarks"][0]["also_in"] == [
        {"provider_id": "amortized-lm-eval", "id": "arc_easy"}
    ]


async def test_endpoint_gets_the_lighteval_copy_and_never_lm_eval(copies):
    loaded = await entries(copies)
    result = catalog.find(loaded, ENDPOINT)
    assert result["available"] == 12  # LightEval's verified entries that don't need logprobs
    # With only lm-eval, nothing can run: the reason is told once, for every non-suite
    # entry, the unverified ones included; suites are not counted.
    lm_eval = [e for e in loaded if e.provider.id == "amortized-lm-eval"]
    nothing = catalog.find(lm_eval, ENDPOINT)
    assert nothing["available"] == 0
    [hidden] = nothing["hidden"]
    assert "tokenizer" in hidden["reason"]
    assert hidden["count"] == sum(1 for r in LM_COMPAT["benchmarks"].values() if not r.get("suite"))
    declared = catalog.Capability.from_model(
        {"endpoint": {"base_url": "http://m/v1"}, "declared_capabilities": ["completions_logprobs"]}
    )
    shared = catalog.find(loaded, declared, name="arc_easy")
    assert _ids(shared["benchmarks"]) == [("amortized-lighteval", "arc:easy", "compatible")]


async def test_when_no_copy_can_run_every_copy_explains_itself(copies):
    result = catalog.find(await entries(copies), ENDPOINT, name="arc easy")
    assert _ids(result["benchmarks"]) == [
        ("amortized-lm-eval", "arc_easy", "incompatible"),
        ("amortized-lighteval", "arc:easy", "incompatible"),
    ]
    assert "tokenizer" in result["benchmarks"][0]["reason"]
    assert "declare" in result["benchmarks"][1]["reason"]


async def test_named_model_needs_a_hugging_face_id_for_lm_eval(copies):
    local = catalog.Capability.from_model({"model_name_or_path": "/models/qwen"})
    result = catalog.find(await entries(copies), local, name="arc_easy")
    assert _ids(result["benchmarks"]) == [("amortized-lighteval", "arc:easy", "compatible")]
    hf = catalog.Capability.from_model({"model_name_or_path": "Qwen/Qwen3-0.6B"})
    assert catalog.find(await entries(copies), hf, name="arc_easy")["benchmarks"][0][
        "provider_id"
    ] == ("amortized-lm-eval")


async def test_lm_eval_recipe_uses_its_own_setting_names(copies):
    loaded = await entries(copies)
    arc = {**definition("arc_easy"), "provider_id": "amortized-lm-eval"}
    recipe, errors = catalog.resolve_recipe(loaded, SERVED, arc, {"max_samples": 20})
    assert errors == []
    assert recipe["parameters"] == {"num_examples": 20, "num_few_shot": 0, "batch_size": 1}
    assert recipe["datasets"] == [
        {
            "task": "arc_easy",
            "hf_repo": "allenai/ai2_arc",
            "hf_subset": "ARC-Easy",
            "splits": ["test"],
            "hf_revision": "not pinned",
        }
    ]
    wrong = {**arc, "parameters": {"max_samples": 20}}
    _, errors = catalog.resolve_recipe(loaded, SERVED, wrong, {})
    assert "max_samples" in errors[0]
    detail = catalog.detail(next(e for e in loaded if e.id == "arc_easy"), SERVED, loaded)
    settings = {s["name"]: s["default"] for s in detail["settings"]}
    assert settings == {"num_examples": "all questions", "num_few_shot": 0, "batch_size": 1}
    assert detail["also_in"] == [{"provider_id": "amortized-lighteval", "id": "arc:easy"}]


async def test_few_shot_default_is_capped_at_the_examples_the_task_holds(both):
    # BBH's chain-of-thought tasks list 3 worked examples; the catalog asks for 5.
    loaded = await entries(both)
    navigate = next(e for e in loaded if e.id == "bbh_cot_fewshot_navigate")
    assert navigate.raw["num_few_shot"] == 5 and navigate.compat["max_few_shot"] == 3
    settings = {s["name"]: s for s in catalog.detail(navigate, SERVED, loaded)["settings"]}
    assert (settings["num_few_shot"]["default"], settings["num_few_shot"]["max"]) == (3, 3)
    bbh = {**definition("bbh_cot_fewshot_navigate"), "provider_id": "amortized-lm-eval"}
    recipe, errors = catalog.resolve_recipe(loaded, SERVED, bbh, {})
    assert errors == [] and recipe["parameters"]["num_few_shot"] == 3
    asked = {**bbh, "parameters": {"num_few_shot": 5}}
    _, errors = catalog.resolve_recipe(loaded, SERVED, asked, {})
    assert errors == ["benchmark settings: num_few_shot: the task has only 3 worked examples"]


def test_benchmark_definitions_validate_each_providers_settings():
    from amortized.evaluation.spec import BenchmarkEvaluation, benchmark_limit

    lm = BenchmarkEvaluation(
        kind="benchmark",
        provider_id="amortized-lm-eval",
        benchmark_id="arc_easy",
        parameters={"num_examples": 5},
    )
    assert lm.parameters == {"num_examples": 5, "batch_size": 1}
    assert benchmark_limit(lm.parameters) == 5
    assert benchmark_limit({"max_samples": 7, "provider": "endpoint"}) == 7
    with pytest.raises(ValueError, match="max_samples"):
        BenchmarkEvaluation(
            kind="benchmark",
            provider_id="amortized-lm-eval",
            benchmark_id="arc_easy",
            parameters={"max_samples": 5},
        )


async def test_hub_job_names_the_tokenizer_for_lm_eval():
    from amortized.integrations.evalhub.preparation import _hf_tokenizer

    repo = MagicMock()
    repo.get_job = AsyncMock(return_value={"config": {"model_name_or_path": "Qwen/Qwen3-0.6B"}})
    assert await _hf_tokenizer({"model_name_or_path": "Qwen/Qwen3-0.6B"}, repo) == "Qwen/Qwen3-0.6B"
    assert await _hf_tokenizer({"training_job_id": "t1"}, repo) == "Qwen/Qwen3-0.6B"
    repo.get_job.assert_awaited_once_with("t1")
    with pytest.raises(JobBuildError, match="not a Hugging Face model id"):
        await _hf_tokenizer({"model_name_or_path": "/models/local"}, repo)


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


async def test_recipe_refuses_an_unpinned_adapter_as_a_server_limit(hub):
    hub.provider["runtime"] = {"k8s": {"Image": "adapter:latest"}}
    recipe, errors = catalog.resolve_recipe(await entries(hub), ENDPOINT, definition(), {})
    assert recipe is None
    assert "pinned adapter image" in errors[0] and SERVER_LIMIT in errors[0]


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


async def test_hub_job_carries_the_cards_metric_as_its_primary_score(hub):
    from evalhub.models import JobSubmissionRequest

    from amortized.api.jobs import _benchmark_view
    from amortized.integrations.evalhub.preparation import benchmark_selection

    # LightEval's catalog names acc_norm for arc:easy; its adapter reports arc:easy.acc.
    recipe, errors = catalog.resolve_recipe(await entries(hub), SERVED, definition("arc:easy"), {})
    assert errors == []
    assert recipe["catalog_entry"]["primary_score"]["metric"] == "acc_norm"
    selected = benchmark_selection(recipe, definition("arc:easy"))
    assert selected["primary_score"] == {"metric": "arc:easy.acc", "lower_is_better": False}
    assert _benchmark_view(recipe, {}).metric == "arc:easy.acc"
    request = JobSubmissionRequest.model_validate(
        {"name": "n", "model": {"url": "http://m/v1", "name": "m"}, "benchmarks": [selected]}
    )
    assert request.benchmarks[0].primary_score.metric == "arc:easy.acc"


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
