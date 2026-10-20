# Proposal: Re-architect agent "gates" — enforce via tools and roles, not user-persona messages

**Status:** RFC / for discussion
**Date:** 2026-10-15
**Scope:** `src/amortized/api/agent.py` gate framework introduced across PRs #495–#502

---

## TL;DR

The guardrails we added to keep a weak driver (GLM-flash) on the rails are **in-band conversational interventions** — hard-coded messages injected into the turn stream (often read by the model as user/system turns) plus **force-correction loops**. They work for weak models but **measurably hobble frontier models (opus-4.8)**, which read them as the human contradicting them and respond by getting cautious and servile.

The fix is not to delete the guardrails — several encode real invariants and real past incidents — but to **change the mechanism**:

1. **Invariants** → enforce **structurally at the tool boundary** (tool preconditions return tool errors; tool availability is role-scoped). Always on, model-agnostic, zero happy-path cost.
2. **Feedback the model must see** → deliver through the model's **native channel** (tool results/errors), or the **developer/system role** — **never as the `user`**, and never persisted into the user transcript.
3. **Fabrication** → fix at the root (UI/state reads from tool outputs, not model prose), so we can delete the detect-and-force-correct machinery entirely.
4. **Chatty scaffolding** that remains → make it **tier-aware** (on for weak drivers, off for frontier models).

Principle: *middleware is fine when it lives in the tool contract and is invisible on the happy path; it is harmful when it impersonates the user or interrupts the model mid-reasoning.*

---

## 1. Background — what we built

The #495–#502 series added three classes of middleware in `api/agent.py`:

### (a) Injected imperatives in the turn stream
`_SUBAGENT_JOB_DONE_PROMPT`, `_SUBAGENT_JOB_FAILED_PROMPT`, `_SUBAGENT_HELP_PROMPT`. These are prefixed `[SYSTEM EVENT]` / `[SUBAGENT NEEDS HELP]` but are **delivered in-band as a turn the model consumes like input**. Example:

> `[SYSTEM EVENT] Your delegated job finished successfully. Your task for this delegation is complete. Your ONLY next action is to call signal_subagent_completion … Do NOT call any other tools … hand control back now.`

The `[SYSTEM EVENT]` prefix is itself the tell: the code *knows* this isn't the user, but there is no real "system says" channel in the loop, so it fakes one inside a message.

### (b) Detect-and-scold gates (with force-correction loops)
- `_apply_provenance_gate` — *"Block + force-correct fabricated identifiers before they reach the user."* Builds a grounding corpus (tool I/O + prior text) and flags any identifier in the model's reply that isn't grounded, then **re-runs the model to correct it** (`MAX_PROVENANCE_RETRIES`).
- `_apply_job_claim_gate` / `_dispatch_provenance_violations` — detects the model claiming a job exists/succeeded (or advancing to a downstream step) when the DB record says otherwise; same force-correct-then-caveat remediation.

### (c) Flow gates
- `_enforce_single_interaction` — orders prompt-review → preview-approval → confirm, re-running the model to clear a held committing card (`MAX_INTERACTION_RERUNS`).

**The origin is explicit in the code comments:** the fabrication gate exists because *"the GLM-5.3-flash run invented a job id."* These are weak-model scaffolds — and they are currently applied to **every** driver, including opus.

---

## 2. Symptom

- Frontier-model (opus-4.8) e2e runs show **hindered capability**: the model hedges, re-confirms, and under-uses tools.
- The asymmetry is the diagnosis: **GLM-flash behaves well with the gates; opus is restrained by them.** A guardrail that helps the weak model and hurts the strong one is a weak-model scaffold applied too broadly.

---

## 3. Root cause — why in-band correction hurts strong models

1. **Role confusion is behavioral, not cosmetic.** Frontier models are trained to treat the `user` role as ground-truth intent and to over-weight the most recent turn. A turn that says *"STOP — you fabricated X, do Y"* — even tagged `[SYSTEM EVENT]` — reads as the human contradicting and micromanaging. The trained response is to become **servile**: more hedging, more confirmation-seeking, less decisive action. That is precisely the reported "restrained capability."
2. **It poisons the transcript.** Injected turns persist. Several turns later the model is reasoning over a conversation in which "the user" repeatedly interrupted with corrections, and it infers a distrustful/confused counterpart — so it dials down initiative **permanently**, not just on the gated turn.
3. **Force-correction loops are the worst offender.** They spend the model's turn re-justifying instead of acting, and on a **false positive** they drag a model that was *right* into defending a non-mistake. Weak models tolerate this (they weren't going to do better); strong models lose real work.
4. **It is self-contradictory.** To stop the model from fabricating provenance, the server fabricates provenance — a user turn that never happened.

---

## 4. The correct pattern

Course-correction should follow this ladder, in order of preference:

### 4.1 Prevent, don't correct
If an action is invalid, make it **unavailable or impossible**, not discouraged.
- **Tool availability scoping** — #502 already does this well (the SDG leaf literally cannot call `validate_eval_job`). Extend the pattern.
- **Tool preconditions** — a tool validates its own invariants and returns a structured error when they're unmet. No conversational steering needed, and behavior is identical for opus and GLM.

### 4.2 If it happens anyway, speak through the tool channel
Tool results/errors are the model's **native** self-correction channel (the ReAct/Reflexion pattern: feedback arrives as an *observation*, not as a person scolding you). A tool error does not corrupt the user relationship and does not trigger the servility reflex. Most current gates react to tool *misuse*, so this is where most belong.

### 4.3 Only if it is not tool-shaped, use the developer/system role
Never `user`; ideally **ephemeral** (system context for that turn, not persisted into the durable user history). Phrase it as **content** ("job X failed"), not an imperative ("YOU MUST call…"). The model API distinguishes roles — use them.

### 4.4 Kill fabrication at the root instead of policing it
The provenance gate solves the wrong problem. The real issue is that the model is allowed to **narrate** state (job ids, "your dataset is ready") that **the UI then trusts**. Fix the source of truth:
- Render ids / status / cards from **tool results**, not model prose.
- Instruct the model (system prompt) to never state an id it didn't receive from a tool.

Then a hallucinated id is cosmetic — nobody downstream believes it — and the detect-and-force-correct machinery can be **deleted**. We stop having to lie to the model to stop the model from lying.

---

## 4b. Validation (2026-10-15) — correction-channel A/B + opencode API facts

Two findings from validating the proposal against the live opencode API and a channel A/B (Opus 4.8 / Sonnet 4.6 / gpt-5.4-mini) revise the mechanism details above:

**opencode exposes a real `system` channel.** `POST /session/{id}/message` accepts a `system` string (and `noReply`) alongside the user `parts` — verified in the deployed OpenAPI and empirically (a `system` instruction was honored verbatim). So steering *can* be delivered as system context rather than a fake user turn. BUT: message **roles** are only `user`/`assistant`, part types are `text`/`tool`/`step-*`, and a `tool` part must bind to a real `tool_use` the model called — **there is no way to inject a free-standing tool-result** for a turn the model didn't initiate. So §6's "deliver the handoff as a tool-result-shaped payload" is **not expressible**; the realistic non-`user` channel is the `system` string.

**No single *message* channel is a clean universal fix** (A/B, N=1 per cell, direction consistent):

| correction via | strong models (Opus/Sonnet) | weak models (gpt-5.4-mini) |
|---|---|---|
| `user` | corrects, but **servile/apologetic**, apology leaks to the user | corrects accurately |
| `system` | **clean**, no apology | **de-prioritized** — kept over-claiming / vague pivot |

- **`user`** → reliable for weak models, but servile on strong models + poisons the transcript (the regression we're chasing).
- **`system`** → clean on strong models, but weak models ignore it (it's advisory) — the models that most need the guardrail skip it.

**Tool errors are the only channel with no disqualifying flaw:** *unignorable* (the call failed — no result to proceed on, so a weak model can't skip it) and *environmental* (a ReAct-style observation, not a person scolding the model — no servility reflex). Confirmed directly: the `generation_type` tool error self-corrected gracefully, and the `not_ready` invariant already lives as a tool 422 (`validate_*`/`create_*` reject a non-`succeeded` parent).

**Consequences for this proposal:**
1. **Make tool preconditions/errors the PRIMARY home** for every check that can be tool-shaped (not just the worked example). Delete the redundant prose gate that re-checks the same invariant.
2. For the **residue that cannot be tool-shaped** (premature-submit prose, stale-reuse disclosure, state-truth prose claims, the subagent handoffs, interaction-ordering steers): **channel-tier it** — `frontier` uses the clean `system` channel (or drops the purely cosmetic ones, since the UI renders from tool outputs); `weak` keeps the `user` channel (the only one it obeys).
3. The §6 handoff migration is **not a free swap** — it moves from `user` to `system` (weak-unreliable), so it must be tiered, not globally switched.

## 4c. Final design (post-review, 2026-10-20): remove, don't tier

Review feedback (eshwarprasadS) pushed the thesis to its conclusion under one principle — **AX = UX**: keep the agent's context equal to what the user sees. The tiering in §5 below is **superseded**; the shipped design *removes* the conversational scaffolding rather than tier-gating it, because once the gates are gone the tier gates nothing:

- **Structural invariants stay at the tool boundary, always-on** — `validate_*`/`create_*` → 422 via `_require_dependency_succeeded` (don't advance past a non-`succeeded` upstream), for every driver. Verified live (blocks queued, passes succeeded; training/eval dual-dep).
- **Provenance gate — deleted** (annotation included): the UI renders ids/status from tool **outputs**, not prose, so a fabricated id is inert; the persona forbids it at the source. The caveat was agent-didn't-author text and never fired.
- **Job-claim gate — deleted** (not tiered): the only correctness piece (upstream-ready) is the tool-boundary 422; "queued"-style over-claims are cosmetic given tool-rendered state; the persona prevents them.
- **Interaction ordering** — the server **no longer strips** stacked cards or re-runs the model. One-interaction-per-turn is enforced by the tool's own `agent_instruction` HALT directive (which outranks the system prompt) + the "ask one question at a time" persona rule; the frontend already renders one card per type (`.find` / last-valid), so a non-compliant driver degrades gracefully, never breaks. This keeps *what the agent emits == what the user sees* (no silent stripping drift).
- **Ordering preconditions (VRAM/data/model before the confirm card) — deleted**: the training/eval **workflows still mandate** those steps ("you MUST show VRAM estimates before model options"), so the coverage lives in the skill, not a backstop gate. (Follow-up #5: also put a server-computed `resource_estimate` *on* the confirm card so the cost is guaranteed-visible at the decision point — a model→size resolver + card render, scoped separately.)
- **Driver-tier flag — deleted**: a static frontier/weak label is a moving maintenance target, and with the scaffolding gone it gated nothing.
- **Subagent hand-back — kept** on the `system` channel (event fact on the user channel), validated live on both a frontier (opus-4.8) and a real weak driver (GLM 5.3).

Net on `agent.py`: **−800 lines**, no gate/correction/stripping layer — the proxy routes, backfills the UI phase, and relays; everything else is the tool boundary + personas + the card.

## 5. The opus-vs-GLM compromise *(superseded by §4c — kept for history)*

Decouple **invariants** from **scaffolding**:

| | Mechanism | When it fires | Cost to a strong model |
|---|---|---|---|
| **Invariants** (don't advance past a failed job; no double-submit; prompt-review before confirm) | tool precondition / tool scoping → structured error | only on an actual violation | ~zero |
| **Scaffolding** (chatty nudges, force-correct loops, "[SYSTEM EVENT] hand back now") | tier-gated: on for weak drivers, off/minimal for frontier | per-turn, preemptive | high (the reported regression) |

So: **guardrails as tool contracts (cheap, universal) + prompting scaffolds (gated to weak drivers).** GLM keeps its training wheels; opus gets a clean (Agent ↔ User ↔ Tools) loop.

---

## 6. Migration plan

For each existing gate: classify it **invariant** vs **nudge**, then move it to the right mechanism.

What shipped (per §4c — everything below is **deleted**, not tier-gated, except where noted):

| Former gate | Class | Shipped outcome |
|---|---|---|
| `_apply_provenance_gate` (+ annotation) | fabrication | **Deleted.** The UI renders ids/status from tool OUTPUTS, not prose, so a fabricated id is inert; a persona no-fabrication rule prevents it at the source. |
| `_apply_job_claim_gate` / `_dispatch_provenance_violations` | invariant + prose-policing | The invariant (don't advance past a non-`succeeded` upstream) is the **tool-boundary 422** (`validate_*`/`create_*` → `_require_dependency_succeeded`), always-on. The prose-claim gate is **deleted** (not tiered). |
| `_enforce_single_interaction` (stripping) + the in-turn re-run | ordering | **Deleted.** One-interaction-per-turn is the tool's `agent_instruction` HALT directive (outranks the system prompt) + the "ask one question at a time" persona rule; the frontend renders one card per type, so a non-compliant driver degrades gracefully. |
| Confirm-card ordering preconditions (`agent_confirm.py`) | ordering | **Deleted.** The VRAM/data/model/judge steps are mandated by the training/eval **workflows**, so coverage lives in the skill. (Follow-up: also put `resource_estimate` on the card.) |
| Driver-tier flag (`agent_driver_tier`) | — | **Deleted.** With the scaffolds gone it gated nothing. |
| `_SUBAGENT_JOB_DONE/FAILED_PROMPT` | handoff | **Kept**, on the **`system` channel** (event fact on the user `text`); reinforced by a persona hand-back rule. |
| `_SUBAGENT_HELP_PROMPT` / `[SUBAGENT COMPLETED]` resume seed | handoff | **Kept**, user-role — these are turn-*drivers* for a programmatic turn (like `[CONTEXT]`/`[RESUMED]`), not reactive corrections. |

Cross-cutting: there is no reactive-correction layer anymore, so no gate re-authors a `user` turn; the only system-channel use is the hand-back imperative. Structural invariants live at the tool boundary and are always-on for every driver.

---

## 7. Worked example — "don't advance past a failed upstream job"

**Today (in-band correction):** the model calls `validate_eval_job` / proceeds while the parent SDG job failed → `_dispatch_provenance_violations` detects it → the server force-corrects the model (re-run), then appends a caveat. Opus, which may have been mid-legitimate-reasoning, gets bounced.

**Proposed (tool precondition):** the invariant lives in the tool. Pseudocode:

```python
# in the create/validate job tool handler
async def create_eval_job(config, ...):
    parent = config.get("parent_job_id")
    if parent:
        row = await repo.get_job(parent)
        if row is None:
            return ToolError(f"parent job {parent} not found — cannot create eval from it")
        if row["status"] != "succeeded":
            return ToolError(
                f"upstream job {parent} is '{row['status']}', not 'succeeded'; "
                f"its dataset is not available. Resolve or re-run it before evaluating."
            )
    ...  # normal creation
```

The model reads the `ToolError` as an **observation** and self-corrects (tell the user, re-run SDG, etc.) with no role fakery, no transcript poisoning, and **identical behavior for opus and GLM**. The server never has to impersonate anyone, and the force-correct loop is gone.

---

## 8. Risks & non-goals

- **Non-goal: deleting the guardrails' intent.** The invariants are legitimate and some encode real incidents (the GLM job-id fabrication was a genuine bug). This is a **re-architecture of mechanism**, not a removal of goals.
- **Risk: regressing weak-driver support.** If GLM is a product requirement (cheaper driver), keep its scaffolds — but behind a tier flag, not globally on. Structural invariants (tools) continue to protect it for free.
- **Risk: tool-error ergonomics.** Tool errors must be phrased as clear, actionable observations; a vague error can confuse a weak model as much as a user-turn did. Treat error copy as prompt surface.
- **Migration order:** start with §7 (highest-value, clearly invariant), then the fabrication root-cause fix (§4.4, likely recovers most of opus's lost capability), then the handoff prompts, then tier the remainder.

---

## 9. Recommendation

Adopt the ladder in §4 as the house rule for all system-initiated course-correction:

> **Prevent via tool availability → fail via tool precondition → (last resort) inform via developer/system role.** The `user` role is reserved for the actual human. The middle layer is invisible on the happy path and speaks in the correct role when it must intervene.

Then execute the §6 migration, beginning with the §7 worked example.
