"""Deterministic matchers (``adjudicate: auto`` rows), as a registry.

Each ``match.type`` maps to a registered function ``(Run, match) -> status``.
Register a new matcher with ``@matcher("name")`` — no central dispatch to edit.
``score_auto`` looks the type up; an unknown type falls back to ``review``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from monitor.scorer.model import Run, _calls
from monitor.scorer.requires import _requires_met

Matcher = Callable[[Run, dict[str, Any]], str]
MATCHERS: dict[str, Matcher] = {}


def matcher(name: str) -> Callable[[Matcher], Matcher]:
    def deco(fn: Matcher) -> Matcher:
        MATCHERS[name] = fn
        return fn

    return deco


_ERROR_MARKERS = ('"errors"', "validation_error", "is required", "invalid", "must be")

# Substrings that identify a base-model name in present_options titles, so the
# model_choice check recognises a model list even when the question itself does
# not contain the word "model". Lowercased family names, not exact ids, so new
# sizes/variants still match.
_MODEL_NAME_MARKERS = (
    "qwen",
    "llama",
    "granite",
    "mistral",
    "gemma",
    "phi-",
    "smol",
    "deepseek",
)


def _is_ok_output(status: str | None, output: str | None) -> bool:
    if status == "error":
        return False
    text = (output or "").lower()
    return not any(marker in text for marker in _ERROR_MARKERS)


def _catalog_names(run: Run) -> set[str]:
    """The set of valid model identifiers the eval judge may name, taken from the
    `name` field of list_models / get_eval_endpoint_suggestions outputs (eval
    workflow: judge model comes from these). Deliberately the `name`, NOT
    `model_name`: config.judge.model is the `name` (e.g. "gpt-oss"), and using the
    `model_name` instead (e.g. "openai/gpt-oss-120b") is exactly the mis-naming
    this guards — a substring test would let it slip, so match `name` exactly."""
    names: set[str] = set()
    for tool in ("list_models", "get_eval_endpoint_suggestions"):
        for c in _calls(run, tool):
            try:
                data = json.loads(c.get("output") or "")
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            for key in ("models", "known_endpoints"):
                for e in data.get(key) or []:
                    if isinstance(e, dict) and e.get("name"):
                        names.add(str(e["name"]))
    return names


def _spec_match(call: dict[str, Any], spec: Any) -> bool:
    """A tool_before endpoint is either a bare tool name (str) or a
    ``{tool, where}`` dict that also matches on logged arg fields (e.g. mode)."""
    if isinstance(spec, str):
        return call.get("tool") == spec
    if call.get("tool") != spec.get("tool"):
        return False
    return all(call.get(k) == v for k, v in (spec.get("where") or {}).items())


def _spec_field_missing(run: Run, spec: Any) -> bool:
    """logging-dependency guard: True when a `where` field is required but no
    call of that tool carries it (log predates the signal -> defer to review).
    False when the tool was never called (let the normal missed/wrong path run)."""
    if isinstance(spec, str) or not spec.get("where"):
        return False
    keys = spec["where"].keys()
    calls = _calls(run, spec.get("tool"))
    return bool(calls) and not any(any(k in c for k in keys) for c in calls)


@matcher("tool_called")
def _m_tool_called(run: Run, match: dict[str, Any]) -> str:
    where = match.get("where") or {}
    tools = match.get("tools") or [match["tool"]]  # `tools` = any-of tool names
    needle = match.get("output_contains")  # optional substring test on tool output
    for c in run.tool_calls:
        if c.get("tool") not in tools:
            continue
        if not all(c.get(k) == v for k, v in where.items()):
            continue
        if match.get("status") and c.get("status") != match["status"]:
            continue
        if needle and needle not in (c.get("output") or ""):
            continue
        return "met"
    # Reuse/resume: the step legitimately did not run because the agent
    # inherited config from a prior job (e.g. SDG reused a prior job's teacher,
    # so there was no teacher selection and hence no pricing to show). The
    # `na_if_reused` probe identifies that prior-config load -> n/a, not missed.
    na = match.get("na_if_reused")
    if na:
        na_tools = na.get("tools") or [na["tool"]]
        na_where = na.get("where") or {}
        na_needle = na.get("output_contains")
        for c in run.tool_calls:
            if c.get("tool") not in na_tools:
                continue
            if not all(c.get(k) == v for k, v in na_where.items()):
                continue
            if na_needle and na_needle not in (c.get("output") or ""):
                continue
            return "n/a"
    return "missed"


@matcher("chained")
def _m_chained(run: Run, match: dict[str, Any]) -> str:
    # A validate_* call carries a non-empty field from `any_of` — used for
    # pipeline wiring (parent_job_id / data_run_id / eval_data_run_id) and
    # for other required config refs (e.g. the eval `judge`). Absent -> missed.
    keys = match.get("any_of") or ["parent_job_id", "data_run_id"]
    for c in _calls(run, match["tool"]):
        if any(c.get(k) for k in keys):
            return "met"
    return "missed"


@matcher("validate_ok")
def _m_validate_ok(run: Run, match: dict[str, Any]) -> str:
    calls = _calls(run, match["tool"])
    if not calls:
        return "missed"
    return "met" if any(_is_ok_output(c.get("status"), c.get("output")) for c in calls) else "wrong"


@matcher("tool_before")
def _m_tool_before(run: Run, match: dict[str, Any]) -> str:
    before_spec, after_spec = match["before"], match["after"]
    if _spec_field_missing(run, before_spec) or _spec_field_missing(run, after_spec):
        return "review"  # log predates the arg the `where` filters on
    before = [i for i, c in enumerate(run.tool_calls) if _spec_match(c, before_spec)]
    after = [i for i, c in enumerate(run.tool_calls) if _spec_match(c, after_spec)]
    if not after:
        return "missed"
    # Resume/reuse: when the earliest `after` call chains to an artifact built
    # in a prior session (its `na_if_after_chained` field is set) and no
    # `before` call precedes it in THIS log, the within-log ordering rule does
    # not apply (e.g. training resumed from a pre-existing SDG dataset). A
    # genuinely out-of-order run with no such chain still falls through to wrong.
    na_field = match.get("na_if_after_chained")
    if na_field:
        first_after = min(after)
        if not any(i < first_after for i in before) and run.tool_calls[first_after].get(na_field):
            return "n/a"
    if not before or min(before) > min(after):
        return "wrong"
    return "met"


@matcher("recovery")
def _m_recovery(run: Run, match: dict[str, Any]) -> str:
    calls = run.tool_calls
    error_idx = next((i for i, c in enumerate(calls) if c.get("status") == "error"), None)
    if error_idx is None:
        return "n/a"  # nothing to recover from
    recovered = any(c.get("status") == "completed" for c in calls[error_idx + 1 :])
    return "met" if recovered else "wrong"


@matcher("completion")
def _m_completion(run: Run, match: dict[str, Any]) -> str:
    if not run.completion:
        return "missed"
    want = match.get("outcome")
    if want and run.outcome != want:
        return "wrong"
    return "met"


@matcher("scores_present")
def _m_scores_present(run: Run, match: dict[str, Any]) -> str:
    # Outcome (not mechanic): the eval job returned usable rubric scores.
    # get_eval_results can succeed as a call yet carry all-null `scores`
    # (judge scored nothing) -> the eval produced no signal. missed when the
    # results were never fetched; wrong when fetched but every score is null.
    calls = _calls(run, match.get("tool", "get_eval_results"))
    if not calls:
        return "missed"
    for c in calls:
        # output is truncated, so match the `"scores": { ... }` block
        # non-greedily up to its closing brace or the truncation edge.
        m = re.search(r'"scores"\s*:\s*\{(.*?)(?:\}|$)', c.get("output") or "", re.S)
        if m and re.search(r":\s*-?\d+(?:\.\d+)?", m.group(1)):
            return "met"  # at least one criterion has a real numeric score
    return "wrong"


@matcher("solo_delegation")
def _m_solo_delegation(run: Run, match: dict[str, Any]) -> str:
    # Every delegating message was ONLY the delegate call (no text, no other
    # tool). `solo` is captured per delegate call; absent -> log predates the
    # signal, defer to review (logging-dependency rule).
    calls = _calls(run, "delegate_to_subagent")
    target = match.get("target")
    if target:
        calls = [c for c in calls if c.get("target") == target]
    if not calls:
        return "missed"
    if not any("solo" in c for c in calls):
        return "review"
    return "met" if all(c.get("solo") for c in calls) else "wrong"


@matcher("no_error_calls")
def _m_no_error_calls(run: Run, match: dict[str, Any]) -> str:
    # Robustness: no tool call ended in an error status (optionally limited
    # to `tools`). Distinct from `recovery`, which only asks whether a later
    # call succeeded after an error — this penalises the error itself.
    tools = match.get("tools")
    for c in run.tool_calls:
        if tools and c.get("tool") not in tools:
            continue
        if c.get("status") == "error":
            return "wrong"
    return "met"


@matcher("prompt_review")
def _m_prompt_review(run: Run, match: dict[str, Any]) -> str:
    # Integrity: task-distillation Step 8 composes a system prompt and must
    # render it with show_prompt before asking the user to approve it (the row
    # is gated on that skill being loaded). Scored structurally, not on the
    # phrasing of the ask: show_prompt called -> met (the prompt was shown,
    # however the agent worded the request); not called but an SDG config was
    # built -> wrong (a prompt was composed and the user never saw it); neither
    # -> n/a (the prompt-composition step was not reached this run).
    if _calls(run, "show_prompt"):
        return "met"
    if _calls(run, "validate_sdg_job"):
        return "wrong"
    return "n/a"


@matcher("eval_phase")
def _m_eval_phase(run: Run, match: dict[str, Any]) -> str:
    # UI progress: the eval stage must signal phase=eval so the bar shows
    # "Evaluation". The observed bug was eval signalling phase=training,
    # leaving the bar stuck on Model Training.
    if not _calls(run, "validate_eval_job"):
        return "n/a"  # no eval stage in this run
    eval_signals = [
        c for c in run.tool_calls if c.get("tool") == "signal_phase" and c.get("role") == "eval"
    ]
    if not eval_signals:
        return "missed"  # eval ran but never signalled a phase
    if not any("phase" in c for c in eval_signals):
        return "review"  # log predates phase capture (logging-dependency)
    return "met" if any(c.get("phase") == "eval" for c in eval_signals) else "wrong"


@matcher("model_choice")
def _m_model_choice(run: Run, match: dict[str, Any]) -> str:
    # The training stage must let the user CHOOSE the base model — present it
    # as options and wait, never auto-pick a recommended default. Checkable
    # via a present_options (training role) whose question or option titles
    # name a model. n/a when there is no training stage; review when the log
    # predates present_options content capture.
    if not _calls(run, "validate_training_job"):
        return "n/a"  # no training stage in this run
    present = [
        c
        for c in run.tool_calls
        if c.get("tool") == "present_options" and c.get("role") == "training"
    ]
    for c in present:
        question = (c.get("question") or "").lower()
        titles = " ".join(c.get("options") or []).lower()
        if "model" in question or any(m in titles for m in _MODEL_NAME_MARKERS):
            return "met"
    # Distinguish "never offered a model choice" from "log predates the
    # option-content capture": if no present_options anywhere carries the
    # question/options fields, we cannot tell -> defer to review.
    if not any(
        "question" in c or "options" in c
        for c in run.tool_calls
        if c.get("tool") == "present_options"
    ):
        return "review"
    return "wrong"  # training ran but never offered a model choice


@matcher("judge_valid")
def _m_judge_valid(run: Run, match: dict[str, Any]) -> str:
    # The eval judge model must be a real model name from the catalog
    # (list_models / get_eval_endpoint_suggestions). We score the LAST
    # validate_eval_job's `judge` — the submitted config — so a run that
    # mis-names the judge then corrects it (as the reference did:
    # openai/gpt-oss-120b -> gpt-oss) scores `met`, while one that leaves an
    # invalid name scores `wrong`. `judge` is promoted at log time, so no
    # config-input logging is needed.
    judges = [c.get("judge") for c in _calls(run, "validate_eval_job") if c.get("judge")]
    if not judges:
        return "missed"  # no judge recorded (eval never validated / judge unset)
    valid = _catalog_names(run)
    if not valid:
        return "review"  # catalog not captured -> can't decide, defer
    return "met" if judges[-1] in valid else "wrong"


def score_auto(run: Run, match: dict[str, Any]) -> str:
    fn = MATCHERS.get(match.get("type", ""))
    if fn is None:
        return "review"
    return fn(run, match)


def score_run(run: Run, checklist: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in checklist["rows"]:
        if not _requires_met(run, row.get("requires")):
            result[row["id"]] = "n/a"  # precondition absent -> row doesn't apply
            continue
        mode = row.get("adjudicate", "human")
        if mode == "auto" and row.get("match"):
            result[row["id"]] = score_auto(run, row["match"])
        else:
            result[row["id"]] = "review"  # human / llm_judge -> offline
    return result
