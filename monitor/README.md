# Monitor — vibe-testing metrics for Morty

Personal, per-user metrics for comparing different LLMs driving Morty on a fixed
use case (the RFE assessor). No shared service — each person's own amortized
deploy writes its own logs, and a script turns them into numbers.

Two efficiency metrics (turns-to-complete, tokens — plus cost and wall-clock)
and per-aspect performance, scored against an **expected-aspect
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
