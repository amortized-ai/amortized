"""A short per-turn POINTER that re-anchors a subagent to its RESIDENT workflow.

The full step-by-step workflow lives resident in each agent's `workflow.md` (= main; no
thinning, no `phases/` slices). A weaker driver (e.g. GLM-5.3-flash) reading a long resident
prompt sometimes substitutes prose for a required transition tool — it narrates "I'll have a
dataset generated" instead of calling `request_orchestrator_help` — or works ahead out of
order. A SHORT pointer on the system channel ("you're in the <agent> workflow; your
instructions are in `## Workflow`; CALL the transition tool, don't narrate it; don't work
ahead") re-anchors the driver each turn without injecting a second copy of the workflow.

Validated at N=6 (memory `pr526-resident-inject-flash`): un-thinned/resident content fixes the
out-of-order formula probe, and resident+pointer holds the eval→SDG handback (6/6, matching the
full-block inject). This is the SLICE-FREE form of that result — the pointer derives the current
section title from the resident `workflow.md` headings themselves (no `phases/` files), and it is
delivered via opencode's message-level `system` field (the PR509 system-channel machinery), never
prepended to the user text.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from amortized.core.recipes import _get_base_dir

# Only leaf workflow subagents get a pointer. The orchestrator drives on its resident
# workflow.md exactly like main (it never signals a phase; its turns are not re-anchored).
_SUBAGENTS = frozenset({"sdg", "training", "eval"})


def _workflow_path(agent: str) -> Path:
    return _get_base_dir() / "agents" / agent / "workflow.md"


def _norm(value: str) -> str:
    """Normalize a signalled step to a comparable slug: drop a leading numeric ordering prefix
    (`03-`) and collapse every run of non-alphanumeric characters (spaces, em dashes,
    underscores, …) to a single hyphen."""
    out = re.sub(r"^\d+[-_.) ]*", "", value.strip().lower())
    return re.sub(r"[^a-z0-9]+", "-", out).strip("-")


# A section's canonical signal slug is authored right in its heading as an HTML-comment marker —
# `### Step 0 — Confirm what to evaluate <!--signal:understand-task-->` — so the slug the agent
# signals (`signal_phase.step`) and the slug the pointer resolves against live in ONE place (the
# resident workflow.md), and a human-readable title can carry a short, stable slug that isn't just
# the slugified title. Headings with no marker fall back to the normalized title.
#
# The marker ALSO carries that step's obligation inline (what the server's stall adjudicator keys
# on), so the signalled vocabulary AND its obligations live in the SAME place the prompt instructs —
# there is no parallel hardcoded `_REQUIRED` map to drift out of sync. Grammar after the slug:
#   <!--signal:validate owed:validate_eval_job-->          owed: → Owed.any_of (≥1 must appear)
#   <!--signal:system-prompt all:show_prompt,present_options-->   all: → Owed.all_of (every one)
#   <!--signal:held-out-request-help owed:request_orchestrator_help solo-->   solo → Owed.solo
# A marker with no obligation suffix is a LENIENT phase (bare prose is a legitimate free-text
# question). `_parse_obligation` reads the suffix; `_workflow_obligations`/`_skill_obligations`
# build the catalog; `tests/test_phase_marker_drift.py` fails loudly if an owed tool is misspelled.
_SIGNAL_MARKER = re.compile(r"<!--\s*signal:\s*([a-z0-9-]+)\s*([^>]*?)\s*-->")


@lru_cache(maxsize=16)
def _sections(agent: str) -> tuple[tuple[str, str], ...]:
    """Ordered `(slug, title)` for each `### ` heading under the resident workflow.md's
    `## Workflow` section. `slug` is the heading's `<!--signal:…-->` marker, else the normalized
    title. `()` when the file is absent or has no Workflow section. Cached: workflow.md is baked at
    build time; tests that write it on the fly call `_sections.cache_clear()`."""
    try:
        text = _workflow_path(agent).read_text(encoding="utf-8")
    except OSError:
        return ()
    out: list[tuple[str, str]] = []
    in_workflow = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            in_workflow = s[3:].strip().lower() == "workflow"
            continue
        if in_workflow and s.startswith("### "):
            raw = s[4:].strip()
            marker = _SIGNAL_MARKER.search(raw)
            title = _SIGNAL_MARKER.sub("", raw).strip()
            out.append((marker.group(1) if marker else _norm(title), title))
    return tuple(out)


def _resolve_title(agent: str, step: str | None) -> str | None:
    """The resident section title to name in the pointer.

    - No step yet (fresh delegation) → the entry (first) section, so a just-delegated subagent
      is pointed at where to start.
    - A signalled step that matches a section heading (exact or prefix, both normalized) → that
      section.
    - A signalled step that matches no heading (e.g. the coarse UI progress vocabulary like
      `gather_requirements`, which is not a section slug) → None, so the pointer stays generic
      rather than naming a WRONG section."""
    sections = _sections(agent)
    if not sections:
        return None
    if not step:
        return sections[0][1]
    target = _norm(step)
    if not target:
        return sections[0][1]
    for slug, title in sections:
        if slug == target or slug.startswith(f"{target}-") or target.startswith(f"{slug}-"):
            return title
    return None


# ── Skill-step resolution (skill steps ride the same signal_phase cursor as workflow phases) ──
# A subagent's gather phase is driven by a resident SKILL.md with a numbered `### Step N — …`
# sequence. A step's slug comes from a `<!--signal:…-->` heading marker — the SAME convention the
# `## Workflow` sections use — and the driver signals it exactly like a workflow phase:
# `signal_phase(step=<skill-step-slug>)` each turn. So ONE cursor (`last_signal_step`) spans both
# workflow sections and the active skill's steps; the agent.py scan records which skill is loaded
# (from the `skill` tool call) to SCOPE the lookup, because step slugs are unique only WITHIN a
# skill (e.g. `teacher-model`/`samples` recur across skills). A legacy `present_options(step="…")`
# stamp still resolves a slug for a skill whose heading carries no marker (none today).
# `tests/test_phase_marker_drift.py` fails loudly if an instructed slug no longer resolves.

_STEP_STAMP_RE = re.compile(
    r"present_options[^\n]*?\bstep\s*[=:]\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)


def _frontmatter_name(text: str) -> str | None:
    """The `name:` from a SKILL.md YAML frontmatter block, or None."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("name:"):
            return line[len("name:") :].strip()
    return None


@lru_cache(maxsize=1)
def _skill_index() -> dict[str, tuple[str, str]]:
    """Map each skill's frontmatter `name` → (agent, skill-dir relative to that agent's `skills/`
    root), by scanning `agents/*/skills/**/SKILL.md`. The `skill` tool call reports the name; this
    resolves it to where the SKILL.md lives."""
    base = _get_base_dir() / "agents"
    index: dict[str, tuple[str, str]] = {}
    for skill_md in base.glob("*/skills/**/SKILL.md"):
        try:
            name = _frontmatter_name(skill_md.read_text(encoding="utf-8"))
        except OSError:
            continue
        if not name:
            continue
        agent = skill_md.relative_to(base).parts[0]
        reldir = skill_md.parent.relative_to(base / agent / "skills").as_posix()
        index[name] = (agent, reldir)
    return index


@lru_cache(maxsize=64)
def _skill_steps(skill_name: str) -> tuple[tuple[str, str], ...]:
    """Ordered `(slug, title)` for each `### ` step heading in a skill's resident SKILL.md that
    carries a slug — a heading `<!--signal:…-->` marker (preferred; the same convention workflow
    sections use) or, failing that, a `present_options(step="…")` stamp in its body. Headings with
    neither (reference sections like `### columns`, or a generate-only step) are skipped. `()` for
    an unknown skill. Cached like `_sections`; tests writing SKILL.md on the fly call
    `_skill_steps.cache_clear()`."""
    entry = _skill_index().get(skill_name)
    if entry is None:
        return ()
    agent, reldir = entry
    path = _get_base_dir() / "agents" / agent / "skills" / reldir / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    out: list[tuple[str, str]] = []
    title: str | None = None
    marker: str | None = None
    body: list[str] = []

    def _flush() -> None:
        if title is None:
            return
        if marker:
            out.append((marker, title))
            return
        stamp = _STEP_STAMP_RE.search("\n".join(body))
        if stamp:
            out.append((_norm(stamp.group(1)), title))

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("### "):
            _flush()
            raw = s[4:].strip()
            found = _SIGNAL_MARKER.search(raw)
            title, marker, body = (
                _SIGNAL_MARKER.sub("", raw).strip(),
                (found.group(1) if found else None),
                [],
            )
        else:
            body.append(line)
    _flush()
    return tuple(out)


def skill_step_title(skill_name: str | None, stamp: str | None) -> str | None:
    """The SKILL.md step title for the slug the agent signalled (`signal_phase.step`), matched on
    the normalized slug (exact or prefix). None when the skill or slug is unknown."""
    if not skill_name or not stamp:
        return None
    target = _norm(stamp)
    if not target:
        return None
    for slug, title in _skill_steps(skill_name):
        if slug == target or slug.startswith(f"{target}-") or target.startswith(f"{slug}-"):
            return title
    return None


def subagent_pointer(
    agent: str | None,
    step: str | None,
    skill: str | None = None,
) -> str | None:
    """A SHORT system-channel pointer re-anchoring a subagent to its RESIDENT instructions.

    Resolves the single signalled `step` against ONE cursor: a `## Workflow` section wins; else,
    when a skill is loaded, one of its `### Step`s (so a skill step is named just like a workflow
    phase); else nothing (generic pointer). The skill clause shows IFF the current step is a skill
    step — once the driver signals `validate`/`complete` it resolves as a workflow section, so the
    clause falls away with no hardcoded post-gather set. None for a non-subagent (the orchestrator)
    or an agent with no workflow.md (no-op, so callers can compose it unconditionally). Delivered
    via opencode's `system` field — a raw, tagged instruction like the other system-channel steers
    (`[SYSTEM EVENT]`, …), NOT wrapped in `<system-reminder>` and never prepended to the user
    text."""
    if not agent or agent not in _SUBAGENTS or not _sections(agent):
        return None
    wf_title = _resolve_title(agent, step)
    skill_title = None if wf_title else skill_step_title(skill, step)
    at_title = wf_title or skill_title
    at = f" — your current step is **{at_title}**" if at_title else ""

    skill_clause = ""
    if skill_title:
        skill_clause = (
            f" You have the **{skill}** skill loaded; its gathering steps are resident in the "
            "SKILL.md. Work that `### Step` sequence IN ORDER — ask ONE question per turn with "
            "`present_options` and do not skip or batch steps."
        )

    return (
        f"[ACTIVE PHASE] You are in the {agent} workflow{at}. Your full step-by-step "
        "instructions are resident in the `## Workflow` section of your system prompt."
        f"{skill_clause} Find your current step there and follow it now, in order. When a step "
        "calls for a transition tool (e.g. `request_orchestrator_help`, "
        "`signal_subagent_completion`), CALL the tool — do NOT describe the action in prose. Do "
        "not work ahead to a later step."
    )


# ── Phase-liveness adjudication (pure library) ────────────────────────────────────────────
#
# A weak driver (GLM-flash) sometimes substitutes PROSE for a required transition — it narrates
# "I'll have the eval set generated" and yields without calling `request_orchestrator_help`, and
# the pipeline stalls (nobody is driving). The resident workflow + per-turn pointer push the driver
# to emit the right tool; this section is the *detector* for when it did not.
#
# It is deliberately a PURE library — a map plus functions over a turn's tool-call set — with NO
# live consumer here. The proxy has no gate/correction layer by design (see
# docs/agent-gates-proposal.md: in-band correction induces servility in frontier models; invariants
# live at the tool boundary). An offline consumer (the monitor) scores each logged turn with
# `adjudicate`; a future live use must stay tier-aware and off the user role.
#
# The model: scan a turn's tool-call multiset and classify it —
#   - stall      : a tool-owed phase yielded on bare prose (no edge at all)
#   - wrong_edge : an interactive card appeared, but not the one the phase owes
#   - not_solo   : a control-transfer hand-off carried text or other tool calls (must be solo)
#   - batching   : >1 distinct step slug signalled in one turn
# Phases absent from the derived catalog (`_required()`) are LENIENT (a genuine free-text question
# is expected). The catalog is SUBAGENT-only (sdg/training/eval); the orchestrator's phase is proxy
# control-flow, not a signalled step, so it has no reliable (role, step) key and is never
# adjudicated. Keys use the resident `<!--signal:slug owed:…-->` vocabulary.

# Tools that transfer control to someone else (so the turn is never a stall) …
_CONTROL_TRANSFER: frozenset[str] = frozenset(
    {"delegate_to_subagent", "request_orchestrator_help", "signal_subagent_completion"}
)
# … and tools that render an interactive card the user acts on.
_CARDS: frozenset[str] = frozenset(
    {
        "present_options",
        "show_prompt",
        "validate_sdg_job",
        "validate_training_job",
        "validate_eval_job",
    }
)
# Any of these means control is accounted for (not a bare-prose stall).
_EDGES: frozenset[str] = _CONTROL_TRANSFER | _CARDS


def emits_card(tools: Collection[str]) -> bool:
    """True when the turn rendered an interactive card the user acts on (present_options /
    show_prompt / validate_*). Such a turn did its owed work, so the nudge consumers must not
    re-drive it: re-driving advances the message boundary past the rendered card and strands it
    (the user sees no card). The sole exception is genuine `batching` — the caller still handles
    that — but a missing `signal_phase`, a stale-key `wrong_edge`, or a `stall` on a card-bearing
    turn is a false drop of a user-actionable card."""
    return bool(_CARDS & set(tools))


@dataclass(frozen=True)
class Owed:
    """What a tool-owed phase must emit in a turn.

    `any_of` — at least one must appear (the common case: a single required edge).
    `all_of` — every one must appear (today only the task-distillation system-prompt review step,
               which owes BOTH `show_prompt` and the approval `present_options`).
    `solo`   — the turn must be ONLY the required tool, no text or other tool calls (a pure
               control-transfer hand-off — e.g. eval's held-out-request-help, whose resident
               section forbids narrating the hand-off).
    """

    any_of: frozenset[str] = frozenset()
    all_of: frozenset[str] = frozenset()
    solo: bool = False


# The tools a phase may legitimately owe — used ONLY as a test-time typo guard (the runtime parser
# is lenient: an unknown token is ignored, never crashes the proxy). It is NOT a second obligation
# catalog — the obligations themselves live solely on the markers — but it IS a tool-name allowlist,
# so to avoid a new hand-maintained list we derive it from the vocabulary the adjudicator already
# keeps: the card + control-transfer edges (`_EDGES`), plus the two owable tools that are neither a
# card nor a control transfer — `skill` (opencode's skill loader) and `clone_sdg_config_for_eval` (a
# plain action the sdg route step may own). `tests/test_phase_marker_drift.py` asserts every tool a
# marker owes is in this set, so a typo fails CI instead of silently making the phase unsatisfiable.
_OWABLE_TOOLS: frozenset[str] = _EDGES | frozenset({"skill", "clone_sdg_config_for_eval"})

# Default obligation for a stepped-skill gather step whose marker declares none: ask one question
# with present_options (preserves "every gather step owes present_options"). A review step overrides
# this by declaring `all:show_prompt,present_options` on its heading marker.
_DEFAULT_SKILL_STEP_OWED = Owed(any_of=frozenset({"present_options"}))


def _parse_obligation(spec: str) -> Owed | None:
    """Parse a signal marker's obligation suffix into an `Owed`, or None for a lenient phase.

    Grammar (whitespace-separated tokens after the slug): `owed:t1,t2` → any_of, `all:t1,t2` →
    all_of, `solo` → solo. An empty / obligation-less suffix is lenient (None)."""
    any_of: frozenset[str] = frozenset()
    all_of: frozenset[str] = frozenset()
    solo = False
    for tok in spec.split():
        if tok == "solo":
            solo = True
        elif tok.startswith("owed:"):
            any_of = frozenset(filter(None, tok[len("owed:") :].split(",")))
        elif tok.startswith("all:"):
            all_of = frozenset(filter(None, tok[len("all:") :].split(",")))
    if not (any_of or all_of or solo):
        return None
    return Owed(any_of=any_of, all_of=all_of, solo=solo)


def _heading_obligation(raw: str) -> tuple[str, Owed] | None:
    """(slug, Owed) for a `### ` heading that declares an obligation in its signal marker, or None
    (no marker, or a marker with no obligation suffix → lenient)."""
    m = _SIGNAL_MARKER.search(raw)
    if not m:
        return None
    owed = _parse_obligation(m.group(2) or "")
    return (m.group(1), owed) if owed is not None else None


@lru_cache(maxsize=16)
def _workflow_obligations(agent: str) -> dict[tuple[str, str], Owed]:
    """`(agent, slug) -> Owed` for each `## Workflow` `### ` heading that declares an obligation in
    its signal marker. The single source for the tool-owed phases (replaces the old hardcoded
    `_REQUIRED`): parsed from the SAME markers the pointer/driver use, so it cannot drift from the
    instructed vocabulary. A heading with no obligation suffix is lenient (absent → `owed_for`
    returns None). Cached like `_sections`."""
    try:
        text = _workflow_path(agent).read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[tuple[str, str], Owed] = {}
    in_workflow = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            in_workflow = s[3:].strip().lower() == "workflow"
            continue
        if in_workflow and s.startswith("### "):
            parsed = _heading_obligation(s[4:].strip())
            if parsed is not None:
                out[(agent, parsed[0])] = parsed[1]
    return out


@lru_cache(maxsize=1)
def _required() -> dict[tuple[str, str], Owed]:
    """The tool-owed phase catalog, derived from every subagent's `## Workflow` markers. A
    (role, slug) NOT here is lenient. The coarse `gather` umbrella declares no obligation on
    purpose — during gathering the driver signals the loaded skill's `### Step` slug, whose
    obligation is resolved by `_skill_owed`."""
    out: dict[tuple[str, str], Owed] = {}
    for agent in _SUBAGENTS:
        out.update(_workflow_obligations(agent))
    return out


@lru_cache(maxsize=64)
def _skill_obligations(skill_name: str) -> dict[str, Owed]:
    """`slug -> Owed` for each of a skill's `### ` step headings that declares an obligation in its
    signal marker (e.g. a review step's `all:show_prompt,present_options`). A step with no declared
    obligation falls back to `_DEFAULT_SKILL_STEP_OWED` in `_skill_owed`. Cached like
    `_skill_steps`."""
    entry = _skill_index().get(skill_name)
    if entry is None:
        return {}
    agent, reldir = entry
    path = _get_base_dir() / "agents" / agent / "skills" / reldir / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, Owed] = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("### "):
            parsed = _heading_obligation(s[4:].strip())
            if parsed is not None:
                out[parsed[0]] = parsed[1]
    return out


def _skill_owed(skill_name: str | None, step_value: str | None) -> Owed | None:
    """The obligation for an active stepped-skill gather step, or None when the slug is not a step
    of the skill. A step's heading marker may declare its obligation (a review step owes BOTH
    `show_prompt` and `present_options` via `all:…`); an undeclared gather step defaults to
    `present_options`."""
    if not skill_name or not step_value:
        return None
    target = _norm(step_value)
    if not target:
        return None
    for slug, _title in _skill_steps(skill_name):
        if slug == target or slug.startswith(f"{target}-") or target.startswith(f"{slug}-"):
            return _skill_obligations(skill_name).get(slug, _DEFAULT_SKILL_STEP_OWED)
    return None


def signal_is_known(role: str, step: str, skill: str | None = None) -> bool:
    """True when the signalled `step` is a slug the phase machinery can key on for this role — a
    tool-owed phase (`_required()`), a `## Workflow` section, or a `### Step` of the loaded skill —
    matched EXACTLY on the normalized slug.

    An unrecognized slug (the driver invented one, e.g. `gather_requirements` instead of the real
    `gather`) keys to no obligation (`owed_for` is lenient) and names no section (the pointer stays
    generic), so the adjudicator and the pointer are both blind to it. For the signalled-guard such
    a turn is therefore EQUIVALENT to a skipped signal — the key it leaves behind is unusable.
    Matching is exact (not the pointer's lenient prefix in `_resolve_title`/`skill_step_title`) on
    purpose: an off-by-prefix slug like `gather-requirements` would prefix-resolve to `gather` for
    the pointer yet still leave the adjudicator mis-keyed, so the guard re-drives it to the exact
    vocabulary."""
    if not step:
        return False
    target = _norm(step)
    if not target:
        return False
    if (role, target) in _required():
        return True
    if role in _SUBAGENTS and any(slug == target for slug, _ in _sections(role)):
        return True
    return bool(skill) and any(slug == target for slug, _ in _skill_steps(skill))


def owed_for(
    role: str,
    phase_step: str,
    *,
    skill: str | None = None,
) -> Owed | None:
    """Resolve what a (role, phase_step) turn owes, or None when the phase is lenient.

    Workflow-phase entries win; otherwise, when a skill is loaded, a signalled slug that names one
    of its `### Step`s owes what that step owes (`_skill_owed`). A slug that is neither a workflow
    phase nor a step of the loaded skill — the coarse `gather` umbrella, a non-stepped skill, or an
    unknown slug — is lenient (None).

    The step is normalized to the canonical slug FIRST — the same `_norm` that `signal_is_known`
    applies — so the two never disagree. Without this, a driver that signals a variant separator
    (`held_out_request_help`) passed the signalled-guard as "known" (which normalizes) yet the raw
    catalog lookup here missed it → lenient → a prose-only hand-off on an owed-solo step escaped
    the nudge entirely (#534 P2b)."""
    step = _norm(phase_step)
    owed = _required().get((role, step))
    if owed is not None:
        return owed
    if skill:
        return _skill_owed(skill, step)
    return None


@dataclass(frozen=True)
class Verdict:
    """Outcome of adjudicating one turn. `kind` is one of: "ok", "stall", "wrong_edge",
    "not_solo", "batching". `owed` is the obligation that was checked (None for a lenient phase),
    for building a diagnostic message."""

    kind: str
    owed: Owed | None = None
    detail: str = ""


_OK = Verdict("ok")


def requires_phase_signal(role: str) -> bool:
    """Subagents must stamp `signal_phase` every turn (it keys the phase); the orchestrator never
    signals (its phase is control-flow). A missing signal on a subagent turn means the phase is
    unattributable — a separate failure, upstream of `adjudicate`, checkable with this."""
    return role in _SUBAGENTS


def adjudicate(
    role: str,
    phase_step: str,
    tools: Collection[str],
    *,
    skill: str | None = None,
    awaiting_job: bool = False,
    has_text: bool = False,
    present_steps: Sequence[str] = (),
) -> Verdict:
    """Classify one turn from its emitted tool-call set.

    `tools` — tool names the turn emitted (the caller should EXCLUDE errored calls). `awaiting_job`
      — a non-terminal job is in scope being awaited (waiting is not a stall). `has_text` — the turn
      also emitted assistant prose, for the `solo` hand-off rule. `present_steps` — the `step` slugs
      of each card this turn, to catch batching (>1 distinct step signalled in one turn).

    Returns a `Verdict`; a lenient phase (or an unknown slug) is always "ok"."""
    tool_set = set(tools)

    # Batching is orthogonal to the owed-edge check and applies in any phase.
    if len({s for s in present_steps if s}) > 1:
        return Verdict("batching", detail="multiple step slugs signalled in one turn")

    owed = owed_for(role, phase_step, skill=skill)
    if owed is None:
        return _OK  # lenient phase: bare prose is a legitimate free-text question

    # Solo hand-off: the turn must be ONLY the required call — no prose, no other tool calls. The
    # mandatory per-turn `signal_phase` is NOT a co-tool (every subagent turn stamps it; requiring
    # its absence would make `solo` unsatisfiable), so it is excluded.
    if owed.solo and (owed.any_of & tool_set):
        extraneous = tool_set - owed.any_of - {"signal_phase"}
        if has_text or extraneous:
            return Verdict("not_solo", owed, "hand-off turn carried text or other tool calls")

    # Obligation satisfied?
    if owed.all_of and owed.all_of <= tool_set:
        return Verdict("ok", owed)
    if owed.any_of & tool_set:
        return Verdict("ok", owed)

    # A control transfer always accounts for control — even when it is not THIS phase's owed edge it
    # is a bail, not a stall. (Whether such a handback is premature is a STATE question for the
    # tool-boundary gate, not here.)
    if _CONTROL_TRANSFER & tool_set:
        return Verdict("ok", owed)

    # A card appeared but not the owed one → wrong edge. No edge at all, and no job awaited → stall.
    if _EDGES & tool_set:
        return Verdict("wrong_edge", owed, "an interactive card appeared but not the owed one")
    if awaiting_job:
        return _OK  # a job is in flight and the agent is waiting on it
    return Verdict("stall", owed, "tool-owed phase yielded with no edge")
