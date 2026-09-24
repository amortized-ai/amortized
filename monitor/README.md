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

## End-to-end testing loop

The one variable is **which LLM drives Morty**; everything else is automatic
logging + offline scoring. Per model you compare, the interactive loop is:

1. **Set the model under test** (the comparison axis). Edit the `"model"` line in
   `containers/morty/opencode.json`, then sync it to the cluster — from the
   `amortized-deploy` repo: `make prompt && make deploy-user USER=xyang` (restarts
   opencode; no server/studio image rebuild needed). The model on the
   orchestrator-role turns is what the report keys the comparison on.
2. **Drive the task in Studio** (`localhost:31180` via the SSH tunnel). Start a
   **new** chat and run the RFE-assessor task end-to-end: describe the assessor →
   let the orchestrator delegate **SDG → training → eval**.
3. **Logging is automatic** — each turn appends a `turn` record to
   `/data/monitor/<session>.jsonl` (tokens, `tool_calls`, `duration_ms`).
4. **Mark the boundary** — click **"Mark complete" → success** (or gave-up) in the
   chat header. This is the only manual signal; it writes the `completion` record
   that defines turns-to-complete / tokens / latency.
5. **Repeat 1–4 per model.** Each run is its own session file.

Then pull the logs once and score them all (below).

## Scoring the logs

Run on the devbox (`ssh -A shiv@169.62.17.147`), from the amortized repo. The
server pod name changes on every deploy, so resolve it; the pod has an init
container so `kubectl cp` needs `-c server`; and `uv run` needs a writable
`UV_CACHE_DIR` (the shared cache isn't writable):

```bash
cd /mnt/4TB/workspace/shiv/xyang/amortized

# 1. pull the monitor logs off the current server pod
SPOD=$(kubectl -n amortized-xyang get pods -o name | grep amortized-server | head -1); SPOD=${SPOD#pod/}
rm -rf ./monitor-logs && mkdir -p ./monitor-logs
kubectl -n amortized-xyang cp -c server "$SPOD:/data/monitor" ./monitor-logs

# 2. score (UV_CACHE_DIR required — the shared uv cache isn't writable by you)
export UV_CACHE_DIR=/mnt/4TB/workspace/shiv/xyang/tmp/uv-cache
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case rfe_assess

# 3. (optional) adjudicate the human/LLM rows
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case rfe_assess \
    --emit-review review.csv          # blank template, one row per review item
#   ...fill the `status` column (met/wrong/missed)...
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case rfe_assess \
    --review review.csv               # merge verdicts into the final metrics
```

`AMORTIZED_MONITOR_LOG_DIR` overrides the log directory. In-cluster it defaults
to `/data/monitor` on the server's persistent PVC. Running the script locally
(`./data/monitor`) needs no `UV_CACHE_DIR` override — that gotcha is
devbox-specific.

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
  "tool_calls": [ { "tool": "validate_training_job", "role": "training", "status": "completed",
                    "output": "…", "target": "sdg",
                    "parent_job_id": "…", "data_run_id": "…", "eval_data_run_id": "…" } ],
  // `target` is set on delegate_to_subagent; `parent_job_id`/`data_run_id`/
  // `eval_data_run_id` are captured from validate_* inputs when non-empty (pipeline chaining).
  "started_at": "…", "finished_at": "…", "duration_ms": 0, "ts": "…" }

// completion
{ "kind": "completion", "session_id": "...", "turn_id": "...",
  "outcome": "success|gave_up", "note": null, "ts": "…" }
```
