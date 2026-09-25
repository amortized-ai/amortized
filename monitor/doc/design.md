# Monitor — design & internals

How the monitor works and how the scorer evaluates a run against the checklist.
For commands, see [`../README.md`](../README.md).

## How it works

1. **Logging (backend).** On each turn completion the agent proxy appends a
   `turn` record to `$AMORTIZED_MONITOR_LOG_DIR/<session_id>.jsonl` (default
   `<data_dir>/monitor` — `/data/monitor` on the server PVC in-cluster,
   `./data/monitor` locally). Each opencode assistant message is attributed to
   exactly one turn (dedup by id across the orchestrator and every subagent
   session), so token totals are correct across delegation and multi-step tool
   loops.

2. **Completion (frontend).** The platform has no workflow-completion signal —
   the orchestrator loops forever offering next steps. So the tester clicks
   **"Mark complete"** in the Studio chat header (success / gave-up), which posts
   a `completion` record. That declares the boundary for turns-to-complete and
   total-tokens.

3. **Processing (offline).** `scripts/process_monitor_logs.py` reads the logs,
   scores each run against the use-case checklist, and prints a per-run report
   plus a per-model comparison table. The headline token metric **excludes
   cache-read tokens** (cheap, dominated by the cached system prompt) — it sums
   input + output + reasoning; the raw log keeps the full breakdown incl. cache.

## Checklist

`use_cases/<name>/checklist.yaml` lists the expected aspects for a run. The
`general` checklist is **use-case agnostic**: its rows are derived from the
platform workflow docs (`agents/*/workflow.md`) — the orchestrator → sdg →
training → eval mechanics that hold for any task Morty is driven through, not
anything specific to a given use case.

A specific use case adds task-grounded rows via an **overlay**: its
`use_cases/<name>/checklist.yaml` sets `extends: general` and lists only the
task-specific rows. The script merges them — all `general` rows apply, plus the
overlay's rows (an overlay row with the same `id` overrides the general one).
Example: `use_cases/rfe_assess` extends `general` and adds
`sdg_skill_is_task_distillation` (RFE is a rubric/structured-eval task, so the
correct SDG sub-skill is `task-distillation`), while `general` only asserts that
*some* sub-skill guide was loaded (`sdg_skill_loaded` / `train_skill_loaded`).

Each row declares how it's judged:

- `auto` — scored deterministically from the log (delegation present, config
  validated, error recovered, run completed, …).
- `llm_judge` — scored by an LLM pass over the transcript (`--llm-judge`, default
  model `claude-opus-4-8` = what Morty runs on); emitted as `review` without the
  flag. Assistant text and full (untruncated) tool output are both logged, so
  grounding and anti-fabrication rows (claim-vs-source) are judged from the log
  alone. The judge returns `met`/`wrong`/`n/a`, or `review` when it lacks evidence.
- `human` — left as `review` for offline human adjudication. No `general` rows
  use this today; it stays available for use cases that need a person in the loop.

Statuses: `met`, `wrong` (did it, wrong), `missed` (should have, didn't — the
default when a signal is absent), `n/a`, `review`.

## End-to-end testing loop

The one variable is **which LLM drives Morty**; everything else is automatic
logging + offline scoring. Per model you compare, the loop is:

1. **Set the model under test** (the comparison axis). Edit the `"model"` line in
   `containers/morty/opencode.json`, then sync it to the cluster from the
   `amortized-deploy` repo (`make prompt && make deploy-user USER=<user>` —
   restarts opencode; no server/studio image rebuild). The model on the
   orchestrator-role turns is what the report keys the comparison on.
2. **Drive the task in Studio.** Start a **new** chat and run the RFE-assessor
   task end-to-end: describe the assessor → let the orchestrator delegate
   **SDG → training → eval**.
3. **Logging is automatic** — each turn appends a `turn` record to
   `/data/monitor/<session>.jsonl` (tokens, `tool_calls`, `duration_ms`).
4. **Mark the boundary** — click **"Mark complete" → success** (or gave-up) in
   the chat header. This is the only manual signal; it writes the `completion`
   record that defines turns-to-complete / tokens / latency.
5. **Repeat 1–4 per model.** Each run is its own session file.

Then pull the logs once and score them all.

## How the script evaluates a run against the checklist

Pure offline: `(logs, checklist) → one status per row per run`. The script only
**reads** the JSONL logs (and your review CSV) — it never calls the cluster for
`auto` rows, and never mutates the logs. It runs in five steps.

### Step 1 — Build a `Run` from each log file

Each `<session>.jsonl` becomes one `Run`: its `turn` records plus the
`completion` record (last one wins), sorted by timestamp. The completion record
sets the **boundary** — `counted_turns` = every turn up to (and including) the
completion timestamp (no completion record → all turns counted). `Run.tool_calls`
is then the flattened, chronological, role-attributed list of every tool call
across `counted_turns`. Efficiency metrics are measured over `counted_turns`:

| metric | how |
|---|---|
| turns-to-complete | number of counted turns (each proxy turn = one user message) |
| tokens (excl. cache) | sum of `input + output + reasoning` across counted turns |
| cost | sum of `cost` |
| wall-clock | last `finished_at` − first `started_at` |
| response latency | per user message, `duration_ms` (server receives message → response ready); reported as avg / max |
| model | the model on orchestrator-role turns (Morty's brain — the comparison axis) |

### Step 2 — Score every checklist row against the `Run`

For each row: if `adjudicate: auto` **and** the row has a `match:`, run the
matcher (a deterministic scan of `tool_calls` / the completion record).
Otherwise (`llm_judge` / `human`) the status is `review`, deferred to Step 4
(or filled in-process by `--llm-judge`).

| matcher | met | wrong | missed / other |
|---|---|---|---|
| `tool_called` (`tool` or `tools` any-of; optional `where` e.g. `role: eval`, and `output_contains` substring e.g. `skills/sdg/`) | a matching call exists | — | `missed` if never called |
| `tool_before` (`before`, `after`; each a bare tool name **or** `{tool, where}` to order by an arg e.g. `mode: preview`) | a `before` call precedes the first `after` call | `after` exists but no earlier `before` | `missed` if `after` never called; `review` if the `where` arg is absent from the log |
| `validate_ok` (tool) | a `validate_*` call whose output isn't an error | only failing calls exist | `missed` if never called |
| `chained` (tool, `any_of` fields) | a `validate_*` call carries a non-empty field (e.g. `parent_job_id`, `judge`) | — | `missed` if none set |
| `scores_present` (tool, default `get_eval_results`) | a call output carries a non-null numeric rubric score | fetched but every score null (`scores_n=0`) | `missed` if never fetched |
| `no_error_calls` (optional `tools` filter) | no matching call ended in `status=error` | some call errored | — |
| `solo_delegation` (optional `target`) | every `delegate_to_subagent` message was the delegate call alone | some delegating message had text/other tools | `missed` if no delegation; `review` if `solo` absent from the log |
| `recovery` | an error-status call followed later by a successful one | error but no later success | `n/a` if no error at all |
| `completion` (outcome) | completion record matches the outcome | outcome differs | `missed` if no completion record |

#### Worked example

**The inputs.** A tiny log (`sess-x.jsonl`), after Step 1 flattens it into
`Run.tool_calls`:

```jsonc
// turn (role=sdg)
{"kind":"turn","role":"sdg","tool_calls":[
   {"tool":"get_model_pricing","role":"sdg","status":"completed"},
   {"tool":"validate_sdg_job","role":"sdg","status":"completed","output":"ok"}]}
// turn (role=training)
{"kind":"turn","role":"training","tool_calls":[
   {"tool":"validate_training_job","role":"training","status":"completed","parent_job_id":"sdg-42"}]}
// completion
{"kind":"completion","outcome":"success"}
```

Four checklist rows (verbatim from `use_cases/general/checklist.yaml`):

```yaml
- id: sdg_pricing_shown        # A
  aspect: B
  adjudicate: auto
  match: { type: tool_called, tools: [get_model_pricing, show_model_pricing], where: { role: sdg } }

- id: chain_training           # B
  aspect: D
  adjudicate: auto
  match: { type: chained, tool: validate_training_job, any_of: [parent_job_id, data_run_id] }

- id: order_training_before_eval   # C
  aspect: C
  adjudicate: auto
  match: { type: tool_before, before: validate_training_job, after: validate_eval_job }

- id: grounding_data           # D
  aspect: E
  adjudicate: llm_judge
```

**How each row flows through the steps.**

*Step 1 — build Run.* Checklist untouched. Produces
`Run.tool_calls = [get_model_pricing, validate_sdg_job, validate_training_job]`
(chronological, role-tagged) + `Run.outcome = "success"`.

*Step 2 — score each row.* `score_run` loops `checklist["rows"]`. For each, the
row's `adjudicate` decides routing and its `match` selects the matcher:

| row | `adjudicate` | matcher reads `match:` → scans log | outcome |
|---|---|---|---|
| **A** `sdg_pricing_shown` | `auto` | `tool_called`: find a call whose tool ∈ {`get_model_pricing`,`show_model_pricing`} and `role==sdg` → finds `get_model_pricing` | met |
| **B** `chain_training` | `auto` | `chained`: find a `validate_training_job` call with `parent_job_id` or `data_run_id` non-empty → `parent_job_id="sdg-42"` | met |
| **C** `order_training_before_eval` | `auto` | `tool_before`: index of `validate_training_job` vs `validate_eval_job` → no `validate_eval_job` in log | missed |
| **D** `grounding_data` | `llm_judge` | not `auto` → skipped (deferred to `--llm-judge` / CSV round-trip) | review |

The row's `match.type` picks the function; `match`'s other keys (`tools`,
`where`, `any_of`, `before`/`after`) are the parameters that function scans the
log with. That's the core of "how the checklist plays a role" — **each row is a
query, the log is the data.**

**Matchers scan the whole run, not one turn.** `Run.tool_calls` concatenates
every counted turn's calls into one flat, timestamp-ordered list, so a matcher
sees the entire trajectory (SDG turn → training turn → eval turn) at once.
Ordering matchers like `tool_before` rely on this. For
`order_training_before_eval` (`before: validate_training_job`,
`after: validate_eval_job`), the matcher collects the indices of each tool in the
flat list and compares the **first** occurrence of each (`min(before) <
min(after)`):

```
index 0: get_model_pricing      (sdg turn)
index 1: validate_sdg_job       (sdg turn)
index 2: validate_training_job  (training turn)   <- first "before" = 2
index 3: validate_eval_job      (eval turn)        <- first "after"  = 3
# 2 < 3  ->  met
```

- **met** — a `validate_training_job` appears before the first `validate_eval_job`.
- **wrong** — `validate_eval_job` exists but no `validate_training_job` precedes it
  (evaluated a model it never trained in this run).
- **missed** — no `validate_eval_job` at all (the eval stage never happened).

Cross-turn order is trustworthy (turns are timestamp-sorted); within a single
turn it follows the logged call order. Calls after the completion boundary are
excluded from the scan.

### Step 3 — Resolve to one of five statuses

`met`, `wrong` (did it, but wrong), `missed`, `n/a`, `review`. **`missed` is the
default when a signal is absent**, so an aspect Morty *should* have exhibited but
skipped scores against it whether or not you noticed — an omission can't hide.

### Step 4 — Fill `review` rows

`llm_judge` rows can be scored in-process with `--llm-judge` (an LLM pass over the
transcript; default model `claude-opus-4-8`). Otherwise `llm_judge` / `human`
rows stay `review` until the CSV round-trip: `--emit-review` writes a blank
template (one row per review item per run), you fill the `status` column, and
`--review` merges those verdicts back in. A filled `--review` CSV overrides the
LLM.

### Step 5 — Aggregate

A per-run block (model, outcome, efficiency, each row's status) and a per-model
comparison table — avg turns, avg tokens (excl. cache), avg cost, avg time, avg
latency, and per-aspect met-rate = `met / (met + wrong + missed)` (`review` and
`n/a` excluded).

> **The one hard constraint:** a row can only be `auto` if its signal was
> **captured at logging time** — the matcher only sees the fields in
> `tool_calls`. That is why chaining (`parent_job_id`) and the eval `judge` had
> to be added to the logger before they could move from `llm_judge` to `auto`.
> If a fact isn't logged, it can't be matched, and the row must fall back to
> `review`.

**Debugging a surprising score.** If a row is `missed` but you know it happened,
check that run's `tool_calls` in the raw JSONL — that shows whether the signal
was actually captured, or the matcher (tool name / output text) needs tuning.

## Log record shapes

```jsonc
// turn
{ "kind": "turn", "session_id": "...", "conversation_root": "...", "turn_id": "...",
  "role": "orchestrator|sdg|training|eval", "model": "...", "provider": "...",
  "tokens": { "input": 0, "output": 0, "reasoning": 0, "cache": 0, "total": 0 },
  "tokens_by_role": { "orchestrator": 0 }, "cost": 0.0,
  "tool_calls": [ { "tool": "validate_training_job", "role": "training", "status": "completed",
                    "output": "…", "target": "sdg", "solo": true, "mode": "preview",
                    "parent_job_id": "…", "data_run_id": "…", "eval_data_run_id": "…",
                    "judge": "…" } ],
  // `output` is the full tool output (not truncated). `target`/`solo` are set on
  // delegate_to_subagent (`solo` = the delegating message was that call alone —
  // clean_delegation); `mode` (preview/create), `parent_job_id`/`data_run_id`/
  // `eval_data_run_id` are captured from validate_* inputs when non-empty; `judge`
  // is the eval judge model name (from validate_eval_job; no api_key).
  "texts": [ { "role": "orchestrator", "text": "…assistant prose this turn…" } ],
  // `texts` = assistant natural-language messages, chronological, for the LLM-judge
  // grounding/anti-fabrication rows (claim checked against tool_calls output).
  "started_at": "…", "finished_at": "…", "duration_ms": 0, "ts": "…" }

// completion
{ "kind": "completion", "session_id": "...", "turn_id": "...",
  "outcome": "success|gave_up", "note": null, "ts": "…" }
```
