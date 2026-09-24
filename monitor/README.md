# Monitor — testing metrics for Morty

Personal, per-user metrics for comparing different LLMs driving Morty on a fixed
use case (the RFE assessor). No shared service — each person's own amortized
deploy writes its own logs, and a script turns them into numbers.

Efficiency metrics (turns-to-complete, tokens — plus cost, wall-clock, and
per-message response latency) and per-aspect performance, scored against an **expected-aspect
checklist** so that behaviors Morty was *supposed* to exhibit but skipped are
counted as misses (an omission can't hide behind "nothing in the log").

## How it works

1. **Logging (backend).** On each turn completion the agent proxy appends a
   `turn` record to `$AMORTIZED_MONITOR_LOG_DIR/<session_id>.jsonl` (default
   `<data_dir>/monitor` — `/data/monitor` on the server PVC in-cluster,
   `./data/monitor` locally). Each opencode assistant message is
   attributed to exactly one turn (dedup by id across the orchestrator and every
   subagent session), so token totals are correct across delegation and
   multi-step tool loops.

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

`use_cases/<name>/checklist.yaml` lists the expected aspects for a run. Each row
declares how it's judged:

- `auto` — scored deterministically from the log (delegation present, config
  validated, error recovered, run completed, …).
- `llm_judge` — scored by an LLM pass (planned; emitted as `review` for now).
- `human` — left as `review` for offline adjudication (right-tool correctness,
  anti-fabrication — a log can't judge these).

Statuses: `met`, `wrong` (did it, wrong), `missed` (should have, didn't — the
default when a signal is absent), `n/a`, `review`.

## Usage

```bash
# 1. Run RFE-assessor sessions in Studio (your own deploy), click "Mark complete".

# 2. Score the logs (in-cluster, copy them out first: kubectl cp <server-pod>:/data/monitor ./logs)
python monitor/scripts/process_monitor_logs.py ./logs --use-case rfe_assess

# 3. (optional) adjudicate the human/LLM rows
python monitor/scripts/process_monitor_logs.py ./logs --use-case rfe_assess \
    --emit-review review.csv          # blank template, one row per review item
#   ...fill the `status` column (met/wrong/missed)...
python monitor/scripts/process_monitor_logs.py ./logs --use-case rfe_assess \
    --review review.csv               # merge verdicts into the final metrics
```

`AMORTIZED_MONITOR_LOG_DIR` overrides the log directory. In-cluster it defaults
to `/data/monitor` on the server's persistent PVC; copy the logs out with
`kubectl cp <server-pod>:/data/monitor ./logs` before scoring.

## What the script computes

The script only ever **reads** the JSONL logs (and your review CSV). It never
calls the cluster or an LLM — the `auto` rows are pure deterministic
log-matching — and never mutates the logs.

**Per-file it builds a `Run`** from the `turn` records plus the `completion`
record (last one wins), sorted by timestamp.

**Completion boundary.** The platform has no "task done" signal, so the
completion record defines the boundary: `counted_turns` = every turn up to (and
including) the completion timestamp. All metrics below are measured over
`counted_turns` only (turns after you clicked "complete" don't count). No
completion record → all turns are counted.

**Efficiency, per run:**

| metric | how |
|---|---|
| turns-to-complete | number of counted turns (each proxy turn = one user message) |
| tokens (excl. cache) | sum of `input + output + reasoning` across counted turns |
| cost | sum of `cost` |
| wall-clock | last `finished_at` − first `started_at` |
| response latency | per user message, `duration_ms` (server receives message → response ready); reported as avg / max |
| model | the model on orchestrator-role turns (Morty's brain — the comparison axis) |

**Performance (checklist).** For each row: `auto` rows run a matcher against the
run's flattened `tool_calls` (concatenated across counted turns, in order);
`human` / `llm_judge` rows are emitted as `review`.

| matcher | met | wrong | missed / other |
|---|---|---|---|
| `tool_called` (tool, optional `where`, e.g. `target: sdg`) | a matching call exists | — | `missed` if never called |
| `validate_ok` (tool) | a `validate_*` call whose output isn't an error | only failing calls exist | `missed` if never called |
| `recovery` | an error-status call followed later by a successful one | error but no later success | `n/a` if no error at all |
| `completion` (outcome) | completion record matches the outcome | outcome differs | `missed` if no completion record |

`missed` is the default when a signal is absent, so an aspect Morty *should* have
exhibited but skipped scores against it whether or not you noticed.

**Output:** a per-run block (model, outcome, efficiency, each row's status) and a
per-model comparison table — avg turns, avg tokens (excl. cache), avg cost, avg
time, avg latency, and per-aspect met-rate = `met / (met + wrong + missed)`
(`review` and `n/a` excluded).

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
  "tool_calls": [ { "tool": "validate_sdg_job", "role": "sdg", "status": "completed",
                    "output": "…", "target": "sdg" } ],
  "started_at": "…", "finished_at": "…", "duration_ms": 0, "ts": "…" }

// completion
{ "kind": "completion", "session_id": "...", "turn_id": "...",
  "outcome": "success|gave_up", "note": null, "ts": "…" }
```
