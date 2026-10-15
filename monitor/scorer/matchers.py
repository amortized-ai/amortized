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

from monitor.scorer.model import Run, _calls, _parse_dt, turn_is_stall
from monitor.scorer.requires import _requires_met, skill_loaded
from monitor.scorer.similarity import (
    SIM_EMBED_HIGH,
    SIM_GRAY,
    SIM_HIGH,
    differs_only_numerically,
    embedding_similarity,
    similarity,
)


def _pair_verdict(a: str, b: str) -> str:
    """Compare two option/question strings -> "same" | "gray" | "distinct".

    Lexical first (token-set Jaccard); when the `--embeddings` tier is enabled, cosine
    over embeddings resolves the lexical gray band and catches paraphrases. A
    numeric-only difference ("1000" vs "2000") is always a distinct choice.
    """
    if differs_only_numerically(a, b):
        return "distinct"
    if similarity(a, b) >= SIM_HIGH:
        return "same"
    esim = embedding_similarity(a, b)  # None when embeddings disabled
    if esim is not None:
        # embeddings enabled -> they decide the uncertain band
        return "same" if esim >= SIM_EMBED_HIGH else "distinct"
    return "gray" if similarity(a, b) >= SIM_GRAY else "distinct"


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


@matcher("skill_loaded")
def _m_skill_loaded(run: Run, match: dict[str, Any]) -> str:
    # A sub-skill was loaded via the native `skill` tool. `skill` = exact name,
    # `role` = sdg|training (any sub-skill for that agent). met/missed.
    return (
        "met"
        if skill_loaded(
            run,
            skill=match.get("skill"),
            role=match.get("role"),
            path_contains=match.get("path_contains"),
        )
        else "missed"
    )


@matcher("action_emitted")
def _m_action_emitted(run: Run, match: dict[str, Any]) -> str:
    # An action-required step culminated in its tool emission (the validate_* card).
    # `emits`/`tools` = the deliverable tool name(s). met if any fired; else missed.
    tools = match.get("emits") or match.get("tools") or []
    for c in run.tool_calls:
        if c.get("tool") in tools:
            return "met"
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


_TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")
_STATUS_RE = re.compile(r'"status"\s*:\s*"([a-z_]+)"', re.IGNORECASE)


def _is_terminal(output: str | None) -> bool:
    """True if a get_job output's STATUS FIELD is terminal (not a substring match,
    so `last_error: a step failed` with status=running is not a false terminal)."""
    if not output:
        return False
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            return str(data.get("status", "")).lower() in _TERMINAL_STATUSES
    except (json.JSONDecodeError, TypeError):
        pass
    m = _STATUS_RE.search(output)
    return bool(m and m.group(1).lower() in _TERMINAL_STATUSES)


def _get_job_polls(run: Run) -> list[tuple[Any, str | None, bool, bool]]:
    """(turn_dt, job_id, is_terminal, had_job_id) for each get_job call, with the
    poll's turn timestamp for ordering against job creation."""
    polls: list[tuple[Any, str | None, bool, bool]] = []
    for turn in run.counted_turns:
        tdt = _parse_dt(turn.get("ts"))
        for c in turn.get("tool_calls") or []:
            if c.get("tool") != "get_job":
                continue
            jid = c.get("job_id")
            polls.append((tdt, jid, _is_terminal(c.get("output")), jid is not None))
    return polls


@matcher("job_created")
def _m_job_created(run: Run, match: dict[str, Any]) -> str:
    # A job of the expected type was actually created (job_created record,
    # correlated to this run via the X-Monitor-Session header the frontend sends).
    # Job creation is human-gated, so it does not appear in the agent tool log —
    # only in these server-side records.
    jt = match.get("job_type")
    jobs = [j for j in run.jobs if not jt or j.get("job_type") == jt]
    return "met" if jobs else "missed"


@matcher("job_monitored")
def _m_job_monitored(run: Run, match: dict[str, Any]) -> str:
    # EACH created job of the type must be polled via get_job — by job id, AFTER it
    # was created — until a terminal status. n/a if nothing created; review if polls
    # exist but none carry a job id (log predates input capture); wrong if any job
    # was polled yet never reached terminal; missed if a created job was never polled.
    jt = match.get("job_type")
    jobs = [j for j in run.jobs if not jt or j.get("job_type") == jt]
    if not jobs:
        return "n/a"
    polls = _get_job_polls(run)
    if polls and not any(p[3] for p in polls):
        return "review"  # get_job called but job id never captured -> can't correlate
    verdicts: list[str] = []
    for job in jobs:
        jid = job.get("job_id")
        created = _parse_dt(job.get("ts"))
        # polls for THIS job that occurred at/after its creation
        relevant = [
            p for p in polls if p[1] == jid and (created is None or p[0] is None or p[0] >= created)
        ]
        if not relevant:
            verdicts.append("missed")  # never polled (or only polled before creation)
        elif any(p[2] for p in relevant):
            verdicts.append("met")
        else:
            verdicts.append("wrong")  # polled but never terminal
    if all(v == "met" for v in verdicts):
        return "met"
    if any(v == "wrong" for v in verdicts):
        return "wrong"
    return "missed"


@matcher("no_stall")
def _m_no_stall(run: Run, match: dict[str, Any]) -> str:
    # The agent announced work but made no tool call, then needed a nudge: a stall
    # turn (promise prose + zero tool calls) followed by a later turn that resumes
    # progress (any tool call) = `wrong` (the user-visible failure). No stall -> met;
    # stalls present but never recovered (e.g. trailing / abandoned) -> review (still
    # surfaced, but captured by completion/outcome rows); no assistant text captured
    # anywhere -> review (logging dependency); no turns -> n/a.
    turns = run.counted_turns
    if not turns:
        return "n/a"
    if not any(t.get("texts") for t in turns):
        return "review"
    stalls = [i for i, t in enumerate(turns) if turn_is_stall(t)]
    if not stalls:
        return "met"
    for i in stalls:
        if any(turns[j].get("tool_calls") for j in range(i + 1, len(turns))):
            return "wrong"  # nudge-recovered stall
    return "review"


def _present_options(run: Run) -> list[dict[str, Any]]:
    return _calls(run, "present_options")


def _option_texts(call: dict[str, Any]) -> list[str]:
    """Option *titles* for a present_options call (the promoted `options` field — the
    choice labels). Deliberately titles only: a shared boilerplate description would
    make genuinely distinct options (different model cards) look near-identical."""
    titles = call.get("options")
    return [str(t) for t in titles] if isinstance(titles, list) else []


def _question_text(call: dict[str, Any]) -> str:
    return str(call.get("question") or "")


@matcher("duplicate_options")
def _m_duplicate_options(run: Run, match: dict[str, Any]) -> str:
    # Deterministic pre-filter for `no_duplicate_options`: within a single
    # present_options prompt, two options worded near-identically. High lexical
    # similarity -> wrong; gray band -> review (the llm_judge row is the semantic
    # backstop); n/a if no prompts; review if prompts exist but options weren't
    # captured (log predates full-input capture).
    calls = _present_options(run)
    if not calls:
        return "n/a"
    checkable = False
    gray = False
    for c in calls:
        texts = _option_texts(c)
        if len(texts) < 2:
            continue
        checkable = True
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                verdict = _pair_verdict(texts[i], texts[j])
                if verdict == "same":
                    return "wrong"
                if verdict == "gray":
                    gray = True
    if not checkable:
        return "review"
    return "review" if gray else "met"


@matcher("repeated_decision")
def _m_repeated_decision(run: Run, match: dict[str, Any]) -> str:
    # Deterministic pre-filter for `no_repeated_decisions`: a present_options prompt
    # that re-asks a near-identical question ALREADY asked *in the same stage*. We
    # segment by delegate_to_subagent so the same question for different stages/
    # datasets (training-set size vs held-out eval-set size — a legitimately fresh
    # decision) is NOT flagged. n/a with <2 prompts; review if questions weren't
    # captured; numeric-only differences are treated as distinct.
    calls = _present_options(run)
    if len(calls) < 2:
        return "n/a"
    # bucket questions by delegation segment, in order
    segments: dict[int, list[str]] = {}
    seg = 0
    captured_any = False
    for c in run.tool_calls:
        if c.get("tool") == "delegate_to_subagent":
            seg += 1
            continue
        if c.get("tool") != "present_options":
            continue
        q = _question_text(c)
        if not q.strip():
            continue
        captured_any = True
        segments.setdefault(seg, []).append(q)
    if not captured_any:
        return "review"
    gray = False
    for questions in segments.values():
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                verdict = _pair_verdict(questions[i], questions[j])
                if verdict == "same":
                    return "wrong"
                if verdict == "gray":
                    gray = True
    return "review" if gray else "met"


@matcher("context_after_switch")
def _m_context_after_switch(run: Run, match: dict[str, Any]) -> str:
    # Deterministic proxy for history retention across a mid-conversation model
    # switch (failure point D): after the switch boundary, a present_options
    # question that re-asks one posed BEFORE the switch is the clearest observable
    # sign context was lost. n/a if no switch or nothing asked after it; review if
    # questions weren't captured; the llm_judge row is the richer backstop.
    idx = run.model_switch_index
    if idx is None:
        return "n/a"
    before_q: list[str] = []
    after_q: list[str] = []
    captured_any = False
    for i, turn in enumerate(run.counted_turns):
        for c in turn.get("tool_calls") or []:
            if c.get("tool") != "present_options":
                continue
            q = _question_text(c)
            if not q.strip():
                continue
            captured_any = True
            (after_q if i >= idx else before_q).append(q)
    if not after_q:
        # either no post-switch prompts, or none captured
        return "review" if not captured_any and _present_options(run) else "n/a"
    if not before_q:
        return "met"  # nothing decided before the switch to re-ask
    gray = False
    for qa in after_q:
        for qb in before_q:
            verdict = _pair_verdict(qa, qb)
            if verdict == "same":
                return "wrong"
            if verdict == "gray":
                gray = True
    return "review" if gray else "met"


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
