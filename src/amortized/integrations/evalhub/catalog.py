"""Benchmark catalog: Eval Hub's provider catalogs joined with each adapter's contract.

The contract (files/evalhub/compat/<provider>.json, written by `make evalhub-contract`) says
what each benchmark needs, read from the task configs in the pinned adapter image, and which
metrics the adapter really reported when the benchmark ran through Eval Hub, including the
one that is its score. Everything here is read-only and deterministic: exact name matching
and fixed-order narrowing questions; no ranking, embeddings or paging.

When two providers ask the same questions (same Hugging Face dataset, subset and split) and
the contract confirms they score them the same way, Morty sees one entry: the copy from the
first provider in settings.evalhub_providers that can run for this model. The other copy is
listed under "also_in" and can still be named.
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
from amortized.evaluation.spec import LightEvalParameters, LMEvalParameters
from amortized.integrations.evalhub import server_limit

SCHEMA = "amortized.evalhub.compat/v2"
# What the release helpers read from the pinned image, before any run (make evalhub-describe).
DESCRIBED = "amortized.evalhub.described/v1"
# present_options renders at most 4 choices (api/ui.py); more become a typed question.
CARD_LIMIT = 4
# The user picks a benchmark, never a framework: each entry carries its provider.
FACETS = ("area", "kind")
QUESTIONS = {
    "area": "Which area should the benchmark cover?",
    "kind": "Which kind of benchmark?",
}
COMPLETIONS_LOGPROBS = "completions_logprobs"
HF_TOKENIZER = "hf_tokenizer"
SETTINGS_PROFILES: dict[str, type[BaseModel]] = {
    "lighteval-endpoint-v1": LightEvalParameters,
    "lm-eval-completions-v1": LMEvalParameters,
}
# What a settings profile's adapter needs from the model, beyond what its scoring needs.
_PROFILE_NEEDS = {
    "lm-eval-completions-v1": (
        HF_TOKENIZER,
        "lm-evaluation-harness sends every question to /v1/completions and loads the"
        " model's Hugging Face tokenizer, which Amortized knows only for a model it serves"
        " from a Hugging Face id. Evaluate a model Amortized serves",
    ),
}
# Categories spelled differently across Eval Hub providers. Used only for narrowing;
# the catalog itself is never rewritten.
KIND_ALIASES = {"coding": "code", "cybersecurity": "security"}
# Recipe keys that decide compatibility but not what a score means.
_CONTEXT_KEYS = ("model_serving", "declared_capabilities")
_SETTING_HELP = {
    "num_few_shot": "Worked examples placed in front of each question",
    "max_samples": "Number of questions to evaluate (unset = all)",
    "num_examples": "Number of questions to evaluate (unset = all)",
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
    profile: str = ""  # the adapter's settings profile, from the compatibility file


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
        return str((score(self.compat) or {}).get("metric") or "")

    @property
    def suite(self) -> bool:
        if self.compat is not None and "suite" in self.compat:
            return bool(self.compat["suite"])
        return "suite" in (self.raw.get("tags") or ())

    def facet(self, facet: str) -> tuple[str, ...]:
        if facet == "area":
            return self.areas or ("",)
        return (self.kind,)


_HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


@dataclass(frozen=True)
class Capability:
    serving: str  # "served": Amortized starts vLLM; "endpoint": an outside URL
    declared: frozenset[str]
    known: frozenset[str] = frozenset()  # facts Amortized knows about a model it serves

    @classmethod
    def from_model(cls, model: dict[str, Any]) -> Capability:
        declared = frozenset(model.get("declared_capabilities") or ())
        if model.get("endpoint"):
            return cls("endpoint", declared)
        # A tuned model's tokenizer is its base model's (training configs name an HF id;
        # the Hub job build checks it); a named model needs an owner/name HF id.
        if model.get("training_job_id") or _HF_ID.match(str(model.get("model_name_or_path"))):
            return cls("served", frozenset(), frozenset({HF_TOKENIZER}))
        if model.get("model_name_or_path"):
            return cls("served", frozenset())
        raise ValueError(
            "Settle the model first: pass training_job_id, model_name_or_path or endpoint"
        )

    @property
    def features(self) -> frozenset[str]:
        if self.serving == "served":
            # vLLM serves /v1/completions with echo and logprobs.
            return frozenset({"chat", COMPLETIONS_LOGPROBS}) | self.known
        return frozenset({"chat"}) | self.declared


def score(compat: dict[str, Any] | None) -> dict[str, Any] | None:
    """The metric that scores an entry, {"metric", "lower_is_better"}, or None.

    The only reader of the score. Hub, the adapter launcher, MLflow and Amortized's results
    all match it by exact name, so only the contract names it: a metric the pinned adapter
    reported when the entry ran through Eval Hub. Never upstream's catalog.
    """
    record = (compat or {}).get("score")
    if not isinstance(record, dict) or not record.get("metric"):
        return None
    return {
        "metric": str(record["metric"]),
        "lower_is_better": bool(record.get("lower_is_better")),
    }


def no_score_reason(compat: dict[str, Any] | None) -> str:
    record = (compat or {}).get("score")
    if isinstance(record, dict) and not record.get("metric"):
        return str(record.get("reason") or "")
    return ""


def recipe_score(recipe: dict[str, Any]) -> dict[str, Any] | None:
    return score(recipe.get("compatibility"))


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


async def load(
    sdk: Any,
    provider_ids: list[str] | None = None,
    described: dict[str, dict[str, Any]] | None = None,
) -> list[Entry]:
    """Read the enabled providers. The SDK's Benchmark model drops domains and
    modalities, so read the documented REST response directly.

    `described` replaces a provider's contract with the helpers' file, for the
    conformance run that the contract is built from.
    """
    entries = []
    for provider_id in provider_ids or config_mod.settings.evalhub_providers:
        data = (await sdk._request_get(f"/evaluations/providers/{provider_id}")).json()
        runtime = (data.get("runtime") or {}).get("k8s") or {}
        # Hub 1.0.3 serializes the Go field as "Image"; provider YAML uses "image".
        image = str(runtime.get("Image") or runtime.get("image") or "")
        compat: dict[str, Any] | None
        if described is not None and provider_id in described:
            compat, problem = described[provider_id], ""
        else:
            compat, problem = load_compat(provider_id)
        if compat is not None and compat.get("adapter_digest") != image_digest(image):
            compat, problem = (
                None,
                (
                    "the compatibility file describes a different adapter image; rebuild it"
                    " with `make evalhub-describe evalhub-conform evalhub-contract`"
                ),
            )
        provider = Provider(
            id=provider_id,
            name=str(data.get("title") or data.get("name") or provider_id),
            image=image,
            catalog_source=dict((compat or {}).get("catalog") or {}),
            problem=problem,
            profile=str((compat or {}).get("settings_profile") or ""),
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
    # What the adapter itself needs holds even for entries the helper could not describe.
    need = _PROFILE_NEEDS.get(
        entry.provider.profile or str((entry.compat or {}).get("settings_profile"))
    )
    if need is not None and need[0] not in capability.features:
        return "incompatible", need[1]
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


def identity(entry: Entry) -> tuple[tuple[str, str, tuple[str, ...]], ...] | None:
    """The questions an entry scores: (canonical HF repo, subset, splits) per dataset.

    The revision is left out: a pinned and an unpinned copy ask the same questions.
    None when the helper could not establish every part.
    """
    compat = entry.compat or {}
    datasets = compat.get("datasets") or []
    if compat.get("status") != "known" or entry.suite or not datasets:
        return None
    if any(not d.get("hf_repo") or not d.get("splits") for d in datasets):
        return None
    return tuple(
        sorted(
            (str(d["hf_repo"]), str(d.get("hf_subset") or "default"), tuple(d["splits"]))
            for d in datasets
        )
    )


def same_questions(entries: list[Entry]) -> list[list[Entry]]:
    """Groups of entries in different providers that ask the same questions.

    A provider with two entries on the same questions scores them differently, so such a
    group is left out.
    """
    by_identity: dict[Any, list[Entry]] = {}
    for entry in entries:
        key = identity(entry)
        if key is not None:
            by_identity.setdefault(key, []).append(entry)
    found = []
    for group in by_identity.values():
        providers = [e.provider.id for e in group]
        if len(set(providers)) > 1 and len(set(providers)) == len(providers):
            found.append(group)
    return found


def _same_scoring(group: list[Entry]) -> bool:
    """Whether the contract confirms each member scores the questions as the others do."""
    for entry in group:
        confirmed = {
            (str(c.get("provider_id")), str(c.get("id")))
            for c in (entry.compat or {}).get("same_scoring") or ()
        }
        if any((o.provider.id, o.id) not in confirmed for o in group if o is not entry):
            return False
    return True


def overlaps(entries: list[Entry]) -> dict[tuple[str, str], list[Entry]]:
    """Copies: the same questions in different providers, scored the same way, in provider
    order. Keyed by every member's (provider_id, id).

    The same questions alone are not enough: lm-eval scores ARC Easy by acc_norm and
    LightEval by acc, so their numbers cannot stand in for each other.
    """
    groups: dict[tuple[str, str], list[Entry]] = {}
    for group in same_questions(entries):
        if _same_scoring(group):
            for entry in group:
                groups[(entry.provider.id, entry.id)] = group
    return groups


def _default(group: list[Entry], statuses: dict[int, tuple[str, str]]) -> Entry:
    """The copy to offer: the first provider's that can run, else the first's."""
    for wanted in ("compatible", "unknown"):
        for entry in group:
            if statuses[id(entry)][0] == wanted:
                return entry
    return group[0]


def _fold(
    entries: list[Entry],
    groups: dict[tuple[str, str], list[Entry]],
    statuses: dict[int, tuple[str, str]],
    *,
    keep_blocked: bool = False,
) -> list[Entry]:
    """One entry per group of copies. With keep_blocked, a group none of whose copies can
    run keeps every copy, so each one's reason (and way forward) is shown."""
    shown: list[Entry] = []
    for entry in entries:
        group = groups.get((entry.provider.id, entry.id))
        chosen = _default(group, statuses) if group else entry
        if keep_blocked and statuses[id(chosen)][0] not in {"compatible", "unknown"}:
            chosen = entry
        if chosen not in shown:
            shown.append(chosen)
    return shown


def _also_in(entry: Entry, groups: dict[tuple[str, str], list[Entry]]) -> list[dict[str, str]]:
    group = groups.get((entry.provider.id, entry.id)) or []
    return [{"provider_id": e.provider.id, "id": e.id} for e in group if e is not entry]


def view(
    entry: Entry,
    capability: Capability,
    groups: dict[tuple[str, str], list[Entry]] | None = None,
) -> dict[str, Any]:
    status, reason = compatibility(entry, capability)
    compat = entry.compat or {}
    also_in = _also_in(entry, groups or {})
    return {
        "provider_id": entry.provider.id,
        "id": entry.id,
        "name": entry.name,
        "description": str(entry.raw.get("description") or "").strip(),
        "metric": entry.metric,
        **({"no_score": no_score_reason(compat)} if no_score_reason(compat) else {}),
        "questions": entry.raw.get("dataset_size") or None,
        "scoring": compat.get("scoring"),
        "answers_per_question": compat.get("answers_per_question"),
        "compatibility": status,
        **({"reason": reason} if reason else {}),
        # Same questions in another framework; offered here because this provider comes
        # first in the enabled order (or the other copy cannot run for this model).
        **({"also_in": also_in} if also_in else {}),
    }


def _label(facet: str, value: str, pool: list[Entry]) -> str:
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
    statuses = {id(e): compatibility(e, capability) for e in entries}
    groups = overlaps(entries)
    if name:
        return _by_name(entries, capability, name, statuses, groups)
    # A shortlist naming one copy of a benchmark also holds its other copies.
    shortlists = [
        {**s, "members": s["members"] | {k for m in s["members"] for k in _keys(groups.get(m))}}
        for s in shortlists or []
    ]
    shown = _fold(entries, groups, statuses)
    # Only entries verified on the pinned image are offered: an unknown one cannot be
    # approved (resolve_recipe). Its own reason is shown when it is looked up by name.
    browsable = [e for e in shown if statuses[id(e)][0] == "compatible"]
    # What this model can't do (e.g. an endpoint without logprobs): told only when nothing
    # can run. Entries not verified and suites are never offered to anyone; not counted.
    hidden = Counter(statuses[id(e)][1] for e in shown if statuses[id(e)][0] == "incompatible")
    picks = dict(picks or {})
    pool = _narrow(browsable, picks, shortlists)
    if answer:
        step = _question(pool, picks, shortlists)
        if step is None:
            # An exact id first: lm-eval arc_easy and LightEval arc:easy normalize alike.
            chosen = [e for e in pool if answer.strip() == e.id] or [
                e for e in pool if _matches(answer, e.id, e.name)
            ]
            if len(chosen) > 1:
                which = "; ".join(
                    f"{e.id} (scored by {view(e, capability, groups)['metric'] or 'no score'})"
                    for e in chosen
                )
                raise ChoiceError(
                    f"{answer!r} matches {len(chosen)} benchmarks: {which}."
                    " Ask the user which one, then send its id."
                )
            if not chosen:
                ids = ", ".join(e.id for e in pool)
                raise ChoiceError(f"{answer!r} is not one of the listed benchmarks: {ids}")
            return {
                "step": "selected",
                "benchmark": view(chosen[0], capability, groups),
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
    benchmarks = [view(e, capability, groups) for e in pool]
    response.update(step="benchmarks", benchmarks=benchmarks)
    if not pool:
        response["display"] = "text"
        response["hidden"] = [{"reason": r, "count": n} for r, n in sorted(hidden.items())]
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


def _keys(group: list[Entry] | None) -> set[tuple[str, str]]:
    return {(e.provider.id, e.id) for e in group or ()}


def _prefixes(provider: Provider) -> set[str]:
    """Ways a user names a framework: its id, its id without "amortized-", its title."""
    names = (provider.id, provider.id.removeprefix("amortized-"), provider.name)
    return {normalize(n) for n in names} - {""}


def _by_name(
    entries: list[Entry],
    capability: Capability,
    name: str,
    statuses: dict[int, tuple[str, str]],
    groups: dict[tuple[str, str], list[Entry]],
) -> dict[str, Any]:
    key = normalize(name)
    names = {id(e): {normalize(e.id), normalize(e.name)} for e in entries}
    # "lighteval arc easy" names the framework too: then only that framework's entry.
    named = [
        e
        for e in entries
        if any(key.startswith(p) and key[len(p) :] in names[id(e)] for p in _prefixes(e.provider))
    ]
    found = named or _fold(
        [e for e in entries if key in names[id(e)]], groups, statuses, keep_blocked=True
    )
    benchmarks = [view(e, capability, groups) for e in found]
    compatible = [b for b in benchmarks if b["compatibility"] == "compatible"]
    if not benchmarks:
        next_step = (
            f"No benchmark is named {name!r}. Say so; do not suggest a substitute. Ask for the"
            " exact name, or browse by calling find_benchmarks without a name."
        )
    elif len(compatible) == 1:
        next_step = "Call get_benchmark with the compatible match's provider_id and id."
    elif compatible:
        next_step = (
            "More than one benchmark matches: list them (id and name) and ask the user which"
            " one, then call get_benchmark with its provider_id and id."
        )
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
            default: Any = default_few_shot(entry)
            available = (entry.compat or {}).get("max_few_shot")
            if isinstance(available, int):
                bounds["max"] = min(bounds.get("max", available), available)
        elif name == getattr(profile, "LIMIT", None):
            default = "all questions"
        else:
            default = field.default
        settings.append({"name": name, **bounds, "default": default, "help": _SETTING_HELP[name]})
    return settings


def default_few_shot(entry: Entry) -> int:
    """The catalog's few-shot count, capped at the worked examples the task config holds."""
    wanted = int(entry.raw.get("num_few_shot") or 0)
    available = (entry.compat or {}).get("max_few_shot")
    return min(wanted, available) if isinstance(available, int) else wanted


def detail(
    entry: Entry, capability: Capability, entries: list[Entry] | None = None
) -> dict[str, Any]:
    compat = entry.compat or {}
    profile = SETTINGS_PROFILES.get(str(compat.get("settings_profile")))
    return {
        **view(entry, capability, overlaps(entries or [entry])),
        "category": entry.raw.get("category"),
        "areas": list(entry.areas),
        # What the pinned adapter reported in its conformance run, not upstream's list.
        "metrics": compat.get("metrics") or [],
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
        return None, server_limit(
            "this benchmark",
            f"Eval Hub's provider {provider_id!r} has no pinned adapter image digest",
        )
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
        parameters["num_few_shot"] = default_few_shot(entry)
    available = compat.get("max_few_shot")
    if isinstance(available, int) and int(parameters["num_few_shot"] or 0) > available:
        return None, [
            f"benchmark settings: num_few_shot: the task has only {available} worked examples"
        ]
    limit = getattr(profile, "LIMIT", "")
    if limit and config.get("max_samples") and not parameters.get(limit):
        parameters[limit] = config["max_samples"]
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
