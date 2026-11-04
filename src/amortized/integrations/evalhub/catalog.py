"""Benchmark catalog: Eval Hub's provider catalog joined with the release helper's file.

The helper (scripts/evalhub/lighteval_compat.py) runs inside the pinned adapter image and
records what each benchmark needs. Everything here is read-only and deterministic: exact
name matching and fixed-order narrowing questions; no ranking, embeddings or paging.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

import amortized.config as config_mod
from amortized.evaluation.spec import LightEvalParameters

SCHEMA = "amortized.evalhub.compat/v1"
# present_options renders at most 4 choices (api/ui.py); more become a typed question.
CARD_LIMIT = 4
FACETS = ("area", "kind", "provider")
QUESTIONS = {
    "area": "Which area should the benchmark cover?",
    "kind": "Which kind of benchmark?",
    "provider": "Which evaluation framework?",
}
COMPLETIONS_LOGPROBS = "completions_logprobs"
SETTINGS_PROFILES: dict[str, type[BaseModel]] = {"lighteval-endpoint-v1": LightEvalParameters}
# Categories spelled differently across Eval Hub providers. Used only for narrowing;
# the catalog itself is never rewritten.
KIND_ALIASES = {"coding": "code", "cybersecurity": "security"}
# Recipe keys that decide compatibility but not what a score means.
_CONTEXT_KEYS = ("model_serving", "declared_capabilities")
_SETTING_HELP = {
    "num_few_shot": "Worked examples placed in front of each question",
    "max_samples": "Number of questions to evaluate (unset = all)",
    "batch_size": "Requests sent to the model at once",
}


class ChoiceError(ValueError):
    """The answer or pick is not one of the offered choices."""


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    image: str
    catalog_source: dict[str, Any]
    problem: str  # why compatibility is unknown for every entry, or ""


@dataclass(frozen=True)
class Entry:
    provider: Provider
    raw: dict[str, Any]
    compat: dict[str, Any] | None

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def name(self) -> str:
        return str(self.raw.get("name") or self.id)

    @property
    def areas(self) -> tuple[str, ...]:
        return tuple(self.raw.get("domains") or ())

    @property
    def kind(self) -> str:
        category = str(self.raw.get("category") or "")
        return KIND_ALIASES.get(category, category)

    @property
    def metric(self) -> str:
        return str((self.raw.get("primary_score") or {}).get("metric") or "")

    @property
    def suite(self) -> bool:
        if self.compat is not None and "suite" in self.compat:
            return bool(self.compat["suite"])
        return "suite" in (self.raw.get("tags") or ())

    def facet(self, facet: str) -> tuple[str, ...]:
        if facet == "area":
            return self.areas or ("",)
        if facet == "kind":
            return (self.kind,)
        return (self.provider.id,)


@dataclass(frozen=True)
class Capability:
    serving: str  # "served": Amortized starts vLLM; "endpoint": an outside URL
    declared: frozenset[str]

    @classmethod
    def from_model(cls, model: dict[str, Any]) -> Capability:
        declared = frozenset(model.get("declared_capabilities") or ())
        if model.get("endpoint"):
            return cls("endpoint", declared)
        if model.get("training_job_id") or model.get("model_name_or_path"):
            return cls("served", frozenset())
        raise ValueError(
            "Settle the model first: pass training_job_id, model_name_or_path or endpoint"
        )

    @property
    def features(self) -> frozenset[str]:
        if self.serving == "served":
            # vLLM serves /v1/completions with echo and logprobs.
            return frozenset({"chat", COMPLETIONS_LOGPROBS})
        return frozenset({"chat"}) | self.declared


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def image_digest(image: str) -> str:
    return image.split("@", 1)[1] if "@sha256:" in image else ""


def load_compat(provider_id: str) -> tuple[dict[str, Any] | None, str]:
    directory = config_mod.settings.evalhub_compat_dir
    if not directory:
        return None, "no compatibility file is configured"
    try:
        data = json.loads((Path(directory) / f"{provider_id}.json").read_text())
    except FileNotFoundError:
        return None, f"no compatibility file for provider {provider_id!r}"
    except ValueError:
        return None, f"the compatibility file for {provider_id!r} is not valid JSON"
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None, f"the compatibility file for {provider_id!r} has an unsupported schema"
    return data, ""


async def load(sdk: Any, provider_ids: list[str] | None = None) -> list[Entry]:
    """Read the enabled providers. The SDK's Benchmark model drops domains and
    modalities, so read the documented REST response directly."""
    entries = []
    for provider_id in provider_ids or config_mod.settings.evalhub_providers:
        data = (await sdk._request_get(f"/evaluations/providers/{provider_id}")).json()
        runtime = (data.get("runtime") or {}).get("k8s") or {}
        # Hub 1.0.3 serializes the Go field as "Image"; provider YAML uses "image".
        image = str(runtime.get("Image") or runtime.get("image") or "")
        compat, problem = load_compat(provider_id)
        if compat is not None and compat.get("adapter_digest") != image_digest(image):
            compat, problem = (
                None,
                (
                    "the compatibility file describes a different adapter image;"
                    " regenerate it with `make evalhub-compat`"
                ),
            )
        provider = Provider(
            id=provider_id,
            name=str(data.get("title") or data.get("name") or provider_id),
            image=image,
            catalog_source=dict((compat or {}).get("catalog") or {}),
            problem=problem,
        )
        records = (compat or {}).get("benchmarks") or {}
        for raw in data.get("benchmarks") or []:
            entries.append(Entry(provider, raw, records.get(raw["id"])))
    return entries


async def load_shortlists(sdk: Any) -> list[dict[str, Any]]:
    """Hub collections, used only as named navigation lists."""
    data = (await sdk._request_get("/evaluations/collections")).json()
    shortlists = []
    for item in data.get("items") or []:
        members = {(b.get("provider_id"), b.get("id")) for b in item.get("benchmarks") or []}
        shortlists.append(
            {
                "id": str((item.get("resource") or {}).get("id") or item.get("name")),
                "name": str(item.get("name") or ""),
                "members": members,
            }
        )
    return shortlists


def compatibility(entry: Entry, capability: Capability) -> tuple[str, str]:
    """(status, reason) with status compatible | incompatible | unknown | unsupported."""
    if "model" not in (entry.raw.get("evaluation_targets") or ()):
        return "unsupported", "it does not evaluate a model"
    if "text" not in (entry.raw.get("modalities") or ()):
        return "unsupported", "it needs non-text input"
    if entry.suite:
        return "unsupported", "suites are not supported yet"
    compat = entry.compat
    if compat is None:
        return "unknown", entry.provider.problem or "the compatibility file has no record for it"
    if compat.get("status") != "known":
        return "unknown", str(compat.get("reason") or "the helper could not establish it")
    profile = compat.get("settings_profile")
    if profile not in SETTINGS_PROFILES:
        return "unknown", f"settings profile {profile!r} is not implemented by this Amortized"
    scoring = compat.get("scoring")
    if scoring == "likelihood" and COMPLETIONS_LOGPROBS not in capability.features:
        return "incompatible", (
            "it uses likelihood scoring, which needs /v1/completions with echo and logprobs."
            " Evaluate a model Amortized serves, or declare that this endpoint supports"
            " completions with logprobs"
        )
    if scoring not in {"generative", "likelihood"}:
        return "unknown", f"unrecognized scoring {scoring!r}"
    return "compatible", ""


def view(entry: Entry, capability: Capability) -> dict[str, Any]:
    status, reason = compatibility(entry, capability)
    compat = entry.compat or {}
    return {
        "provider_id": entry.provider.id,
        "id": entry.id,
        "name": entry.name,
        "description": str(entry.raw.get("description") or "").strip(),
        "metric": entry.metric,
        "questions": entry.raw.get("dataset_size") or None,
        "scoring": compat.get("scoring"),
        "answers_per_question": compat.get("answers_per_question"),
        "compatibility": status,
        **({"reason": reason} if reason else {}),
    }


def _label(facet: str, value: str, pool: list[Entry]) -> str:
    if facet == "provider":
        return next((e.provider.name for e in pool if e.provider.id == value), value)
    if not value:
        return "Other"
    return value.replace("_and_", " & ").replace("_", " ").capitalize()


def _choices(facet: str, pool: list[Entry]) -> list[dict[str, Any]]:
    counts = Counter(v for e in pool for v in set(e.facet(facet)))
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], _label(facet, kv[0], pool)))
    return [
        {"key": facet, "value": v, "label": _label(facet, v, pool), "count": n} for v, n in ordered
    ]


def _splits(choices: list[dict[str, Any]], size: int) -> bool:
    return len(choices) >= 2 and any(c["count"] < size for c in choices)


def _matches(answer: str, *candidates: str) -> bool:
    key = normalize(re.sub(r"\(\d+\)\s*$", "", answer))
    return bool(key) and key in {normalize(c) for c in candidates if c}


def _narrow(
    pool: list[Entry], picks: dict[str, str], shortlists: list[dict[str, Any]]
) -> list[Entry]:
    unknown = set(picks) - {"shortlist", *FACETS}
    if unknown:
        raise ChoiceError(f"Unknown picks {sorted(unknown)}; send back the picks you received")
    if "shortlist" in picks:
        chosen = next((s for s in shortlists if s["id"] == picks["shortlist"]), None)
        if chosen is None:
            raise ChoiceError(f"No shortlist {picks['shortlist']!r}")
        pool = [e for e in pool if (e.provider.id, e.id) in chosen["members"]]
    for facet in FACETS:
        if facet not in picks:
            continue
        allowed = _choices(facet, pool)
        if picks[facet] not in {c["value"] for c in allowed}:
            labels = ", ".join(c["label"] for c in allowed)
            raise ChoiceError(f"{picks[facet]!r} is not a {facet} here; choose one of: {labels}")
        pool = [e for e in pool if picks[facet] in e.facet(facet)]
    return pool


def _question(
    pool: list[Entry], picks: dict[str, str], shortlists: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]] | None:
    if len(pool) <= CARD_LIMIT:
        return None
    for facet in FACETS:
        if facet in picks:
            continue
        choices = _choices(facet, pool)
        if _splits(choices, len(pool)):
            if not picks:
                for s in shortlists:
                    count = sum((e.provider.id, e.id) in s["members"] for e in pool)
                    if count:
                        label = f"Shortlist: {s['name'] or s['id']}"
                        choices.append(
                            {"key": "shortlist", "value": s["id"], "label": label, "count": count}
                        )
            return facet, choices
    return None


_REPLY = (
    "When the user answers, call find_benchmarks again with the same model, picks set to"
    " this response's picks, and answer set to the user's reply."
)


def find(
    entries: list[Entry],
    capability: Capability,
    *,
    name: str = "",
    picks: dict[str, str] | None = None,
    answer: str = "",
    shortlists: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    shortlists = shortlists or []
    if name:
        return _by_name(entries, capability, name)
    statuses = {id(e): compatibility(e, capability) for e in entries}
    browsable = [e for e in entries if statuses[id(e)][0] in {"compatible", "unknown"}]
    hidden = Counter(
        statuses[id(e)][1] for e in entries if statuses[id(e)][0] in {"incompatible", "unsupported"}
    )
    picks = dict(picks or {})
    pool = _narrow(browsable, picks, shortlists)
    if answer:
        step = _question(pool, picks, shortlists)
        if step is None:
            chosen = [e for e in pool if _matches(answer, e.id, e.name)]
            if len(chosen) != 1:
                ids = ", ".join(e.id for e in pool)
                raise ChoiceError(f"{answer!r} is not one of the listed benchmarks: {ids}")
            return {
                "step": "selected",
                "benchmark": view(chosen[0], capability),
                "next": (
                    "Call get_benchmark with this provider_id and id and the same model,"
                    " then ask the user only about its settings."
                ),
            }
        facet, choices = step
        hit = next((c for c in choices if _matches(answer, c["value"], c["label"])), None)
        if hit is None:
            labels = ", ".join(c["label"] for c in choices)
            raise ChoiceError(
                f"{answer!r} is not one of the choices; ask again: {labels}."
                " (To look a benchmark up by name instead, call find_benchmarks with name.)"
            )
        picks[hit["key"]] = hit["value"]
        pool = _narrow(browsable, picks, shortlists)
    response: dict[str, Any] = {
        "picks": picks,
        "available": len(pool),
        "hidden": [{"reason": r, "count": n} for r, n in sorted(hidden.items())],
    }
    step = _question(pool, picks, shortlists)
    if step is not None:
        facet, choices = step
        response.update(step="question", facet=facet, question=QUESTIONS[facet])
        if len(choices) <= CARD_LIMIT:
            response["display"] = "card"
            response["options"] = [
                {
                    "title": c["label"],
                    "description": f"{c['count']} benchmark{'s' if c['count'] != 1 else ''}",
                    "value": c["label"],
                }
                for c in choices
            ]
            response["next"] = (
                "Call present_options with step 'choose-benchmark', this question and these"
                " options exactly as given. " + _REPLY
            )
        else:
            response["display"] = "text"
            response["choices"] = [{"label": c["label"], "count": c["count"]} for c in choices]
            response["next"] = (
                f"There are {len(choices)} choices, more than a card holds: list every choice"
                " with its count in your message and ask the user to type one. Do not use"
                " present_options for this question. " + _REPLY
            )
        return response
    benchmarks = [view(e, capability) for e in pool]
    response.update(step="benchmarks", benchmarks=benchmarks)
    if not pool:
        response["display"] = "text"
        response["next"] = "No benchmark can run for this model; explain the hidden reasons."
    elif len(pool) <= CARD_LIMIT:
        response["display"] = "card"
        response["options"] = [
            {
                "title": b["id"],
                "description": " · ".join(
                    x
                    for x in (
                        b["name"],
                        b["metric"],
                        f"{b['questions']} questions" if b["questions"] else "",
                        "compatibility unknown" if b["compatibility"] == "unknown" else "",
                    )
                    if x
                ),
                "value": b["id"],
            }
            for b in benchmarks
        ]
        response["next"] = (
            "Call present_options with step 'choose-benchmark' and these options exactly as"
            " given. " + _REPLY
        )
    else:
        response["display"] = "text"
        response["next"] = (
            "List every benchmark (id and name) in your message and ask the user to name one."
            " Do not use present_options. " + _REPLY
        )
    return response


def _by_name(entries: list[Entry], capability: Capability, name: str) -> dict[str, Any]:
    prefixes = {normalize(p) for e in entries for p in (e.provider.id, e.provider.name)}
    key = normalize(name)
    keys = {key} | {key[len(p) :] for p in prefixes if p and key.startswith(p)}
    found = [e for e in entries if keys & {normalize(e.id), normalize(e.name)}]
    benchmarks = [view(e, capability) for e in found]
    if not benchmarks:
        next_step = (
            f"No benchmark is named {name!r}. Say so; do not suggest a substitute. Ask for the"
            " exact name, or browse by calling find_benchmarks without a name."
        )
    elif any(b["compatibility"] == "compatible" for b in benchmarks):
        next_step = "Call get_benchmark with the compatible match's provider_id and id."
    else:
        next_step = (
            "Explain the reason it cannot run as requested and the ways forward it names."
            " Do not offer a different benchmark in its place."
        )
    return {"step": "matches", "benchmarks": benchmarks, "next": next_step}


def _settings(entry: Entry, profile: type[BaseModel]) -> list[dict[str, Any]]:
    settings = []
    for name, field in profile.model_fields.items():
        if name not in _SETTING_HELP:
            continue
        bounds = {}
        for item in field.metadata:
            if getattr(item, "ge", None) is not None:
                bounds["min"] = item.ge
            if getattr(item, "gt", None) is not None:
                bounds["min"] = item.gt + 1  # integer settings
            if getattr(item, "le", None) is not None:
                bounds["max"] = item.le
        if name == "num_few_shot":
            default: Any = int(entry.raw.get("num_few_shot") or 0)
        elif name == "max_samples":
            default = "all questions"
        else:
            default = field.default
        settings.append({"name": name, **bounds, "default": default, "help": _SETTING_HELP[name]})
    return settings


def detail(entry: Entry, capability: Capability) -> dict[str, Any]:
    compat = entry.compat or {}
    profile = SETTINGS_PROFILES.get(str(compat.get("settings_profile")))
    return {
        **view(entry, capability),
        "category": entry.raw.get("category"),
        "areas": list(entry.areas),
        "metrics": entry.raw.get("metrics") or [],
        "tasks": compat.get("tasks") or [],
        "datasets": [
            {**d, "hf_revision": d.get("hf_revision") or "not pinned"}
            for d in compat.get("datasets") or []
        ],
        "settings": _settings(entry, profile) if profile else [],
    }


def resolve_recipe(
    entries: list[Entry],
    capability: Capability,
    definition: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    provider_id, benchmark_id = definition["provider_id"], definition["benchmark_id"]
    entry = next(
        (e for e in entries if e.provider.id == provider_id and e.id == benchmark_id), None
    )
    if entry is None:
        return None, [f"benchmark {benchmark_id!r} is not in provider {provider_id!r}"]
    if "@sha256:" not in entry.provider.image:
        return None, [f"provider {provider_id!r} must use a pinned adapter image digest"]
    status, reason = compatibility(entry, capability)
    if status == "unsupported":
        return None, [f"benchmark {benchmark_id!r} cannot run: {reason}"]
    if status == "incompatible":
        return None, [f"benchmark {benchmark_id!r} is incompatible with this model: {reason}"]
    if status == "unknown":
        return None, [
            f"benchmark {benchmark_id!r} has unknown compatibility ({reason});"
            " it must be resolved before it can be approved"
        ]
    compat = entry.compat or {}
    profile = SETTINGS_PROFILES[str(compat["settings_profile"])]
    parameters = dict(definition.get("parameters") or {})
    if parameters.get("num_few_shot") is None:
        parameters["num_few_shot"] = int(entry.raw.get("num_few_shot") or 0)
    if config.get("max_samples") and not parameters.get("max_samples"):
        parameters["max_samples"] = config["max_samples"]
    try:
        parameters = profile.model_validate(parameters).model_dump(exclude_none=True)
    except ValidationError as exc:
        return None, [f"benchmark settings: {e['loc'][0]}: {e['msg']}" for e in exc.errors()]
    recipe = {
        "version": 1,
        "provider_id": provider_id,
        "benchmark_id": benchmark_id,
        "catalog": entry.provider.catalog_source,
        "catalog_entry": entry.raw,
        "adapter_image": entry.provider.image,
        "adapter_digest": image_digest(entry.provider.image),
        "compatibility": compat,
        "parameters": parameters,
        "datasets": [
            {**d, "hf_revision": d.get("hf_revision") or "not pinned"}
            for d in compat.get("datasets") or []
        ],
        "model_serving": capability.serving,
        "declared_capabilities": sorted(capability.declared),
    }
    return recipe, []


def fingerprint(recipe: dict[str, Any]) -> str:
    """Hash of what decides a score; serving context is recorded but not hashed."""
    scored = {k: v for k, v in recipe.items() if k not in _CONTEXT_KEYS}
    payload = json.dumps(scored, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
