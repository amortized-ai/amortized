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

`use_cases/<name>/checklist.yaml` lists the expected aspects for a run. The
`general` checklist is **use-case agnostic**: its rows are derived from the
platform workflow docs (`agents/*/workflow.md`) — the orchestrator → sdg →
training → eval mechanics that hold for any task Morty is driven through, not
anything specific to a given use case. A specific use case can add task-grounded
rows in its own `use_cases/<name>/checklist.yaml` later.

Each row declares how it's judged:

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
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case general

# 3. (optional) adjudicate the human/LLM rows
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case general \
    --emit-review review.csv          # blank template, one row per review item
#   ...fill the `status` column (met/wrong/missed)...
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case general \
    --review review.csv               # merge verdicts into the final metrics
```

`AMORTIZED_MONITOR_LOG_DIR` overrides the log directory. In-cluster it defaults
to `/data/monitor` on the server's persistent PVC. Running the script locally
(`./data/monitor`) needs no `UV_CACHE_DIR` override — that gotcha is
devbox-specific.

## How the script evaluates a run against the checklist

Pure offline: `(logs, checklist) → one status per row per run`. The script only
**reads** the JSONL logs (and your review CSV) — it never calls the cluster or an
LLM for `auto` rows, and never mutates the logs. It runs in five steps.

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
Otherwise (`llm_judge` / `human`) the status is `review`, deferred to Step 4.

| matcher | met | wrong | missed / other |
|---|---|---|---|
| `tool_called` (`tool` or `tools` any-of; optional `where`, e.g. `target: sdg` or `role: eval`) | a matching call exists | — | `missed` if never called |
| `tool_before` (`before`, `after`) | a `before` call precedes the first `after` call | `after` exists but no earlier `before` | `missed` if `after` never called |
| `validate_ok` (tool) | a `validate_*` call whose output isn't an error | only failing calls exist | `missed` if never called |
| `chained` (tool, `any_of` fields) | a `validate_*` call carries a non-empty field (e.g. `parent_job_id`, `judge`) | — | `missed` if none set |
| `recovery` | an error-status call followed later by a successful one | error but no later success | `n/a` if no error at all |
| `completion` (outcome) | completion record matches the outcome | outcome differs | `missed` if no completion record |

**Worked example — each row is a query, the log is the data.** Take a run whose
`Run.tool_calls` (after Step 1) is:

```
get_model_pricing (role=sdg) | validate_sdg_job (role=sdg) | validate_training_job (role=training, parent_job_id="sdg-42")
```

with `Run.outcome = "success"` and no eval calls. Four checklist rows score
against it like this — the row's `match.type` picks the matcher, the other
`match` keys are the parameters it scans the log with:

| row (`aspect`) | `adjudicate` | `match` → what it scans for | result |
|---|---|---|---|
| `sdg_pricing_shown` (B) | `auto` | `tool_called`: a call with tool ∈ {`get_model_pricing`,`show_model_pricing`} **and** `role=sdg` → finds `get_model_pricing` | **met** |
| `chain_training` (D) | `auto` | `chained`: a `validate_training_job` call with `parent_job_id` **or** `data_run_id` non-empty → `parent_job_id="sdg-42"` | **met** |
| `order_training_before_eval` (C) | `auto` | `tool_before`: needs a `validate_eval_job` after `validate_training_job` → **no `validate_eval_job` in the log** | **missed** |
| `grounding_data` (E) | `human` | not `auto` → not scanned | **review** |

`order_training_before_eval` is **missed** — not skipped — precisely because the
row pre-exists: the checklist declared eval was expected, so its absence from the
log is scored (an omission can't hide). See Step 3.

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

### Step 4 — Fill `review` rows offline

`llm_judge` / `human` rows stay `review` until the CSV round-trip:
`--emit-review` writes a blank template (one row per review item per run), you
fill the `status` column, and `--review` merges those verdicts back in.

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
                    "output": "…", "target": "sdg",
                    "parent_job_id": "…", "data_run_id": "…", "eval_data_run_id": "…",
                    "judge": "…" } ],
  // `target` is set on delegate_to_subagent; `parent_job_id`/`data_run_id`/
  // `eval_data_run_id` are captured from validate_* inputs when non-empty (pipeline
  // chaining); `judge` is the eval judge model name (from validate_eval_job; no api_key).
  "started_at": "…", "finished_at": "…", "duration_ms": 0, "ts": "…" }

// completion
{ "kind": "completion", "session_id": "...", "turn_id": "...",
  "outcome": "success|gave_up", "note": null, "ts": "…" }
```
