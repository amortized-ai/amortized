# Monitor — testing metrics for Morty

Personal, per-user metrics for comparing different LLMs driving Morty on a fixed
use case (the RFE assessor). No shared service — each person's own amortized
deploy writes its own logs, and a script turns them into numbers: efficiency
(turns-to-complete, tokens excl. cache, cost, wall-clock, per-message latency)
and per-aspect performance scored against an expected-aspect checklist.

> **Run the pipeline on the kind cluster.** Logging happens inside the deployed
> server, and the LLM-judge needs the same Vertex access Morty uses, so both the
> runs and the scoring are done against your kind deploy — not on a laptop. The
> commands below assume a shell on the devbox that hosts the cluster.

For how it all works (logging, checklist, scoring internals, log shapes) see
[`doc/design.md`](doc/design.md).

## Usage

Placeholders: `<devbox>` (cluster host), `<amortized-repo>` (repo checkout),
`<namespace>` (your amortized namespace), `<user>` (your deploy user),
`<uv-cache>` (a writable uv cache dir).

### 1. Run the task (per model you compare)

Set the model under test in `containers/morty/opencode.json`, sync it from the
`amortized-deploy` repo (`make prompt && make deploy-user USER=<user>`), then in
Studio start a **new** chat and drive the RFE-assessor task end-to-end
(SDG → training → eval). Click **"Mark complete" → success** (or gave-up) in the
chat header — the only manual signal. Repeat per model; each run is its own
session file. (Details in [`doc/design.md`](doc/design.md#end-to-end-testing-loop).)

### 2. Pull the logs and score them

```bash
ssh -A <devbox>
cd <amortized-repo>

# pull the monitor logs off the current server pod (name changes each deploy;
# the pod has an init container, so `kubectl cp` needs `-c server`)
SPOD=$(kubectl -n <namespace> get pods -o name | grep amortized-server | head -1); SPOD=${SPOD#pod/}
rm -rf ./monitor-logs && mkdir -p ./monitor-logs
kubectl -n <namespace> cp -c server "$SPOD:/data/monitor" ./monitor-logs

# score (UV_CACHE_DIR must point at a writable dir — the shared cache isn't)
export UV_CACHE_DIR=<uv-cache>
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case general
```

### 3. (optional) Score the `llm_judge` rows with an LLM pass

Default model `claude-opus-4-8` (what Morty runs on). Needs the `monitor` extra
(`anthropic`) and Vertex creds in the env — point `ANTHROPIC_VERTEX_PROJECT_ID`
at the **same Vertex project Morty uses** (from the `opencode-llm` secret), plus
`CLOUD_ML_REGION`; or set `ANTHROPIC_API_KEY` for the direct API.

```bash
export ANTHROPIC_VERTEX_PROJECT_ID=<VERTEX_PROJECT>   # must have access to the model
export CLOUD_ML_REGION=<VERTEX_REGION>
uv run --extra monitor python monitor/scripts/process_monitor_logs.py ./monitor-logs \
    --use-case general --llm-judge    # also prints an "LLM-judge rationale" section
```

### 4. (optional) Adjudicate by hand

A filled `--review` CSV overrides the LLM.

```bash
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case general \
    --emit-review review.csv          # blank template, one row per review item
#   ...fill the `status` column (met/wrong/missed)...
uv run python monitor/scripts/process_monitor_logs.py ./monitor-logs --use-case general \
    --review review.csv               # merge verdicts into the final metrics
```

Use `--use-case rfe_assess` for the RFE overlay. `AMORTIZED_MONITOR_LOG_DIR`
overrides the log directory (defaults to `/data/monitor` on the server PVC
in-cluster). Running the script against a local `./data/monitor` needs no
`UV_CACHE_DIR` override — that gotcha is devbox-specific.
