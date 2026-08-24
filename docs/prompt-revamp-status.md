# Prompt Revamp: Session Summary

Branch: `fix/prompt-revamp`

## What We Did

### 1. Fixed "e is not iterable" Runtime Error

Studio's `fetchSessionMessages()` calls `GET /agent/session/{id}/message`.
The proxy was returning a static stub (`{info: {}, parts: []}`) instead of
forwarding to OpenCode. Studio expected an array of messages and crashed
when iterating the wrong shape.

**Fix:** Proxy now forwards GET requests to the active OpenCode session
(subagent or orchestrator) and returns the real response.

### 2. Fixed Orchestrator Not Loading (Wrong Agent)

The proxy was dropping the `agent` field when forwarding messages to the
orchestrator's OpenCode session. OpenCode defaulted to whatever agent it
chose, often `training` instead of `morty`.

**Fix:** Proxy now explicitly passes `agent="morty"` on both the
orchestrator path and the resume-after-completion path.

### 3. Fixed Delegation Detection

OpenCode's POST response only contains `step-start`/`step-finish` parts —
tool invocations happen asynchronously via MCP and don't appear in the
response. The proxy was scanning response parts for `delegate_to_subagent`
tool calls and never finding them.

**Fix:** Replaced response-part scanning with shared in-process state.
The MCP `delegate_to_subagent` and `signal_subagent_completion` endpoints
now write to a pending queue (`_pending_delegations`, `_pending_completions`)
in `agent.py`. The proxy reads and clears the queue after each OpenCode
call. No response parsing needed.

### 4. Scoped SDG Subagent Identity

The SDG subagent was describing itself as capable of training and ML
workflows when asked "who are you?". Added explicit instruction to only
describe SDG capabilities.

### 5. Prevented present_options After Job Submission

The subagent was calling `present_options` immediately after submitting a
job, overlapping with the job monitor card. Added explicit rule to both
subagent prompts: no `present_options` after submission — wait for the
`[SYSTEM EVENT]`.

---

## What's Left: Divergences from Main

Compared all prompt/skill files between `main` and `fix/prompt-revamp`.
Items grouped by severity.

### Critical (Breaks Studio Features)

| # | Item | Detail |
|---|------|--------|
| 1 | **`signal_phase` missing** | Neither subagent prompt mentions it. Studio's progress bar (`PlanProgress`) relies on `signal_phase` calls with the step mapping: `understand_task → load_skill → gather_requirements → estimate_cost → confirm → execute → review`. Progress bar is dead. |
| 2 | **Tool catalog deleted** | `capabilities.md` had every MCP tool with parameters. Deleted entirely. Subagents may not discover tools like `get_artifact_content`, `convert_document`, `cancel_job`, `get_document_chunks`, or know parameter signatures (e.g. `estimate_training_resources` takes `model_size`, `method`, `num_gpus`). |

### High (Degrades Quality)

| # | Item | Detail |
|---|------|--------|
| 3 | **Job status event handling** | Old prompt had specific instructions per status — running ("acknowledge, do NOT call present_options"), succeeded (congratulate, present next steps), failed/cancelled (view logs / try again / start fresh). The orchestrator has thin coverage; subagents have none. |
| 4 | **Student model candidate sizes** | Old prompt listed exact sizes: `0.8B, 2B, 4B, 9B`. New training prompt says "each candidate model size" without specifying which. |
| 5 | **Failed job debugging workflow** | Old had 3 steps: `get_job_detail` → `get_job_logs` → diagnose and suggest fixes. Missing entirely from all prompts. |
| 6 | **Platform readiness validation** | Old explicitly called `get_config` + `list_models` before submission. New vaguely says "silently verify the platform can execute the job" without specifying which tools to call. |
| 7 | **Post-SDG data preview** | Old prompt: after SDG success, call `get_dataset_samples` with `mlflow_run_id` to preview 2-3 sample QA pairs for quality verification. Missing from new flow. |

### Medium (Loses Helpful Detail)

| # | Item | Detail |
|---|------|--------|
| 8 | **Teacher model search strategy** | Old had 6 detailed steps for broadening searches (strip dates, version suffixes, provider prefixes, pick best match). Condensed in new prompt. |
| 9 | **SDG `validate_sdg_job` parameter reference** | Old listed columns, model_configs, processors, document_ids, num_records, topic, mode inline. New relies entirely on sub-skill guides. |
| 10 | **Numeric input guidance** | Old: "For numeric inputs like 'how many samples', suggest 2-3 common values as options." Lost from identity.md. |
| 11 | **Confirmation formatting prohibitions** | Old had explicit list: no tables, no bullet lists, no "key highlights", no parameter summaries, no "click confirm" instructions. New has a shorter version. |
| 12 | **Job detail inspection** | Old: "Call `get_job_detail`, show detailed markdown TABLE with ALL configuration. Do NOT include next-step options — the UI handles navigation." Missing. |

---

## Commits on This Branch (This Session)

| Commit | Description |
|--------|-------------|
| `a449e32` | Guard against missing parts array in agent proxy responses |
| `f9fc145` | Proxy GET session messages to OpenCode instead of returning stub |
| `f4e13b1` | Revert defensive fallbacks (no longer needed) |
| `89a3db1` | Pass `agent=morty` when forwarding to orchestrator session |
| `f75789e` | Add debug logging for tool parts |
| `e9b2a54` | Log full response parts for debugging |
| `c22f96c` | Handle varying tool part formats in detection |
| `e41e9fe` | Match Studio's tool detection (no completion status filtering) |
| `d041f81` | Use shared state for delegation instead of response part scanning |
| `cc7aec2` | Scope SDG subagent identity to SDG-only capabilities |
| `7041f04` | Prevent present_options after job submission |
