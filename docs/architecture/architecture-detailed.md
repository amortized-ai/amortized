# Amortized — Detailed Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            ENTRY POINTS                                  │
│                                                                          │
│  Browser ──────> Studio SPA (:8080)                                      │
│  CLI ───────────────────────────────> Amortized Server (:8000)           │
│  AI Agent (MCP client) ──────────────> MCP endpoint (/mcp)               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Studio (React SPA + Nginx)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Studio — React SPA (port 8080)                                      │
│                                                                      │
│  ┌────────────────┐  ┌─────────────────┐  ┌───────────────────────┐ │
│  │  Job Dashboard  │  │  Recipe Browser  │  │   Cost Calculator    │ │
│  │  - list/detail  │  │  - browse/submit │  │   - SDG/train/eval   │ │
│  │  - logs/cancel  │  │  - overrides     │  │   - model comparison │ │
│  └───────┬────────┘  └────────┬─────────┘  └──────────┬───────────┘ │
│          │                    │                        │             │
│  ┌───────┴────────────────────┴────────────────────────┴───────────┐ │
│  │                    Chat Interface (Morty)                        │ │
│  │                    - conversational ML assistant                 │ │
│  └─────────────────────────────┬───────────────────────────────────┘ │
│                                │                                     │
│  ┌─────────────────────────────┴───────────────────────────────────┐ │
│  │                      Nginx Reverse Proxy                        │ │
│  │                                                                 │ │
│  │  /              -->  Static files (try_files $uri /index.html)  │ │
│  │  /api/          -->  Amortized Server   ($BACKEND_HOST:8000)    │ │
│  │  /agent/        -->  Agent Service      ($AGENT_HOST:8001)      │ │
│  │  /mlflow/       -->  MLflow Server      ($MLFLOW_HOST:5000)     │ │
│  │  /mcp           -->  MCP endpoint       (SSE, no buffering)     │ │
│  │                                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Agent Service — Morty

```
┌──────────────────────────────────────────────────────────────────┐
│  Agent Service (port 8001)                                       │
│                                                                  │
│  HTTP API:                                                       │
│    POST /session             - create new session                │
│    POST /session/:id/message - send message, get response        │
│    GET  /api/health          - health check                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Claude Agent SDK (Opus 4)                                │    │
│  │                                                           │    │
│  │  Prompts (system prompt):                                  │    │
│  │    identity.md     - identity, guardrails, no code/shell  │    │
│  │    capabilities.md - skill manifest, MCP tool catalog     │    │
│  │    workflow.md     - generic workflow, cost rules          │    │
│  │  Skills (loaded on demand from skills/ directory):        │    │
│  │    sdg/guidance.md      -> sub-skill guides               │    │
│  │    training/guidance.md -> sub-skill guides               │    │
│  │    eval/guidance.md     -> sub-skill guides               │    │
│  │                                                           │    │
│  │  MCP Connections:                                         │    │
│  │    amortized  (HTTP)  --> /mcp on Amortized Server        │    │
│  │    mlflow     (SSE)   --> MLflow MCP Server                │    │
│  │                                                           │    │
│  │  Providers: Anthropic API / Vertex AI / Bedrock / Foundry │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Flow:
  Chat UI --> POST /session/:id/message
           --> Claude SDK processes with identity + capabilities + workflow prompts
           --> skill guides loaded on demand from skills/ directory
           --> Calls MCP tools (list_jobs, submit_recipe_job, etc.)
           --> Returns response parts (text + tool results)
```

---

## Amortized Server (FastAPI)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Amortized Server — FastAPI (port 8000)                                      │
│                                                                              │
│  MIDDLEWARE                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  CORS (configurable origins)  -->  API Key Auth (Bearer token)       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  API LAYER (/api/v1)                                                         │
│  ┌─────────────────────────┬──────────────────────┬──────────────────────┐   │
│  │  Jobs Router            │  Recipes Router      │  Costs Router        │   │
│  │                         │                      │                      │   │
│  │  POST   /jobs         │  GET  /recipes       │  POST /costs/sdg     │   │
│  │  GET    /jobs         │  GET  /recipes/:name │  POST /costs/sdg/    │   │
│  │  GET    /jobs/:id     │  PUT  /recipes/:name │       compare        │   │
│  │  DELETE /jobs/:id     │  POST /jobs/recipe   │  POST /costs/training│   │
│  │  GET    /jobs/:id/logs│                      │  POST /costs/training│   │
│  │  GET    /jobs/:id/    │                      │       /method        │   │
│  │         artifacts     │                      │  POST /costs/eval    │   │
│  ├─────────────────────────┼──────────────────────┼──────────────────────┤   │
│  │  Documents Router       │  Models Router       │  Health/Config       │   │
│  │                         │                      │                      │   │
│  │  POST /documents/       │  GET /models         │  GET /health         │   │
│  │       convert           │  (from AI Gateway)   │  GET /config         │   │
│  │  POST /documents/       │                      │                      │   │
│  │       convert/url       │                      │                      │   │
│  │  GET  /documents        │                      │                      │   │
│  │  GET  /documents/:id/   │                      │                      │   │
│  │       content           │                      │                      │   │
│  ├─────────────────────────┼──────────────────────┼──────────────────────┤   │
│  │  MCP Server             │                      │                      │   │
│  │                         │                      │                      │   │
│  │  /mcp                   │                      │                      │   │
│  │  (auto-exposes all      │                      │                      │   │
│  │   endpoints as MCP      │                      │                      │   │
│  │   tools via             │                      │                      │   │
│  │   fastapi-mcp)          │                      │                      │   │
│  └─────────────────────────┴──────────────────────┴──────────────────────┘   │
│                                                                              │
│  CORE DOMAIN LOGIC                                                           │
│  ┌──────────────────┬──────────────────┬──────────────────────────────────┐  │
│  │  core/jobs.py    │  core/compute.py │  core/recipes.py                 │  │
│  │  Job lifecycle:  │  Backend         │  YAML loading with extends:      │  │
│  │  create, get,    │  registry,       │  inheritance, dot-notation       │  │
│  │  list, cancel    │  capability      │  overrides, flatten to config    │  │
│  │                  │  checks          │                                  │  │
│  ├──────────────────┼──────────────────┼──────────────────────────────────┤  │
│  │  core/config_    │  core/judge_     │  core/redact.py                  │  │
│  │  translator.py   │  templates.py    │  Strip api_key, token,           │  │
│  │                  │  Load from       │  password from configs            │  │
│  │  Translates to:  │  templates/eval/ │  and credential patterns         │  │
│  │  - training-hub  │                  │  from log text                   │  │
│  │  - TRL           │                  │                                  │  │
│  │  - asynth synth  │                  │                                  │  │
│  │  - asynth judge  │                  │                                  │  │
│  └──────────────────┴──────────────────┴──────────────────────────────────┘  │
│                                                                              │
│  DOCUMENT PROCESSING (proxy to docling-serve)                                │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  api/documents.py                                                    │    │
│  │                                                                      │    │
│  │  Upload/URL --> docling-serve --> parsed content --> MLflow artifact  │    │
│  │                                                                      │    │
│  │  Formats: markdown, text, JSON, HTML                                 │    │
│  │  Options: OCR toggle, OCR engine (easyocr/tesseract), table mode     │    │
│  │  Storage: MLflow experiment "amortized/documents"                     │    │
│  │    - Source file archived as artifact                                 │    │
│  │    - Parsed content stored as artifact                                │    │
│  │  Security: SSRF protection (blocks private IPs, metadata endpoints,  │    │
│  │    .local/.internal hostnames), 100 MB upload limit                   │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  BACKGROUND WORKER                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  worker_loop() — polls every 2s                                      │    │
│  │                                                                      │    │
│  │  1. pick_pending_job()     FIFO from SQLite (oldest queued)          │    │
│  │  2. Resolve parent         MLflow run --> S3 artifact URI            │    │
│  │     artifacts              (for SDG->Train->Eval chaining)           │    │
│  │  3. Config translate       Job config --> tool-native YAML           │    │
│  │  4. Build JobSpec          cmd, env, config_files, s3_downloads,     │    │
│  │                            resources (GPU count, memory)             │    │
│  │  5. Submit to backend      --> BackendHandle                         │    │
│  │  6. Poll status            backend.status() every 2s                 │    │
│  │  7. On completion:                                                   │    │
│  │     - Extract mlflow_run_id from container logs                      │    │
│  │     - Tag MLflow run with job metadata                               │    │
│  │     - Register model in MLflow Model Registry (training jobs)        │    │
│  │     - Update job status in SQLite                                    │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  DATABASE LAYER                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  SQLite (aiosqlite, WAL mode)                                        │    │
│  │                                                                      │    │
│  │  Repository pattern — raw SQL, no ORM                                │    │
│  │                                                                      │    │
│  │  jobs table:                                                         │    │
│  │    id, type, status, config (JSON), recipe, user_id,                 │    │
│  │    k8s_job_name, k8s_namespace, mlflow_run_id, mlflow_experiment,    │    │
│  │    parent_job_id, error, created_at, started_at, completed_at,       │    │
│  │    backend_handle (serialized JSON)                                  │    │
│  │                                                                      │    │
│  │  Indexes: status, type, created_at, user_id                          │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Compute Backends

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Compute Backends — all implement ComputeBackend Protocol                    │
│                                                                              │
│  Protocol: submit(JobSpec) -> BackendHandle                                  │
│            status(handle)  -> BackendStatus                                  │
│            cancel(handle)  -> None                                           │
│            logs(handle)    -> AsyncIterator[str]                             │
│                                                                              │
│  ┌────────────────────────┬──────────────────────┬─────────────────────────┐ │
│  │  Kubernetes            │  SSH                  │  Local                  │ │
│  │  (kubernetes_asyncio)  │  (asyncssh)           │  (subprocess)           │ │
│  │                        │                       │                         │ │
│  │  Per-job creates:      │  Two modes:           │  - Popen with env       │ │
│  │  - K8s Secret          │  - Container:         │  - venv PATH injection  │ │
│  │    (env vars)          │    podman/docker       │  - Container paths      │ │
│  │  - ConfigMap           │    --gpus all          │    remapped to local    │ │
│  │    (config files)      │    --network host      │  - In-memory process    │ │
│  │  - Job resource        │    volume mounts       │    tracking             │ │
│  │    (pod spec)          │    podman secrets      │                         │ │
│  │  - Init containers     │  - Bare metal:         │  Capabilities:          │ │
│  │    (S3 download        │    nohup + log files   │    LOG_STREAM, STOP     │ │
│  │     via aws-cli)       │    env exports         │                         │ │
│  │  - GPU scheduling      │                       │                         │ │
│  │    (nvidia.com/gpu)    │  Capabilities:         │                         │ │
│  │  - RuntimeClass:       │    GPU, LOG_STREAM,    │                         │ │
│  │    nvidia              │    STOP                │                         │ │
│  │  - ownerReferences     │                       │                         │ │
│  │    (auto-cleanup)      │                       │                         │ │
│  │  - TTL: 3600s          │                       │                         │ │
│  │                        │                       │                         │ │
│  │  Capabilities:         │                       │                         │ │
│  │    GPU, LOG_STREAM,    │                       │                         │ │
│  │    STOP                │                       │                         │ │
│  └────────────────────────┴──────────────────────┴─────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## ML Tool Containers

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ML Tool Containers                                                          │
│                                                                              │
│  ┌──────────────────────────────────┬───────────────────────────────────────┐│
│  │  asynth                          │  training-hub / TRL                    ││
│  │  ghcr.io/amortized-ai/asynth     │  ghcr.io/amortized-ai/training        ││
│  │                                  │                                       ││
│  │  SDG:                            │  Command:                             ││
│  │    asynth synthesize             │    thub <algo> --config config.yaml    ││
│  │    --config synth_config.yaml    │    trl <algo> --config config.yaml     ││
│  │                                  │                                       ││
│  │  Eval:                           │  Algorithms:                          ││
│  │    asynth judge                  │    sft, lora_sft, osft,               ││
│  │    --config config.yaml          │    dpo, grpo, lora_grpo,              ││
│  │    --data eval_data.jsonl        │    kto, gkd, gepa                     ││
│  │                                  │                                       ││
│  │  Inputs:                         │  Inputs:                              ││
│  │    - Teacher model (via Gateway) │    - Training data (from S3 via       ││
│  │    - Task description/seed data  │      init container download)         ││
│  │                                  │    - Base model (HuggingFace)         ││
│  │  Outputs:                        │                                       ││
│  │    - generated_data.jsonl        │  Outputs:                             ││
│  │    - eval scores                 │    - LoRA adapter / merged model      ││
│  │    --> logged to MLflow --> S3   │    - Metrics (loss, eval_loss)        ││
│  │                                  │    --> auto-logged to MLflow (TRL)     ││
│  └──────────────────────────────────┴───────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Services

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Infrastructure Services (plugged in, not bundled — AD-1)                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  MLflow                                                              │    │
│  │                                                                      │    │
│  │  Tracking Server          Model Registry          AI Gateway         │    │
│  │  - Experiments            - Registered models     - Endpoint routing  │    │
│  │  - Runs + metrics         - Model versions        - Routes to LLMs:  │    │
│  │  - Artifact logging       - Stage promotion       │                  │    │
│  │  - Tags (job metadata)    (auto-registered by     │  OpenAI          │    │
│  │                            worker on training     │  Anthropic        │    │
│  │  API used:                 job completion)        │  Google/Vertex    │    │
│  │  - GET  /runs/get                                 │  OpenRouter       │    │
│  │  - POST /runs/set-tag                             │  (live pricing)   │    │
│  │  - POST /registered-models/create                 │                  │    │
│  │  - POST /model-versions/create                    │                  │    │
│  │  - GET  /gateway/endpoints/list                   │                  │    │
│  └────────────────────────────────┬──────────────────┘                  │    │
│                                   │                                      │    │
│                                   v                                      │    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  S3 / MinIO                                                          │    │
│  │                                                                      │    │
│  │  MLflow artifact store — all artifacts flow through MLflow (AD-3)    │    │
│  │  - Generated datasets (SDG output)                                   │    │
│  │  - Trained models / LoRA adapters                                    │    │
│  │  - Training metrics and logs                                         │    │
│  │  - Eval results                                                      │    │
│  │                                                                      │    │
│  │  Access:                                                             │    │
│  │  - MLflow writes via artifact logging                                │    │
│  │  - K8s init containers read via `aws s3 cp/sync`                     │    │
│  │  - Env: MLFLOW_S3_ENDPOINT_URL (for MinIO)                           │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Docling-Serve (optional)                                            │    │
│  │                                                                      │    │
│  │  Document processing service for ingesting PDFs, DOCX, HTML, etc.   │    │
│  │  Amortized proxies requests to docling-serve and stores results      │    │
│  │  in MLflow as artifacts under the "amortized/documents" experiment.  │    │
│  │                                                                      │    │
│  │  Access:                                                             │    │
│  │  - Amortized calls docling-serve /v1/convert/file and /source        │    │
│  │  - Parsed content + source files archived in MLflow                  │    │
│  │  - Env: AMORTIZED_DOCLING_URL                                        │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Template Library

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Template Library                                                            │
│                                                                              │
│  templates/sdg/            10 strategies                                     │
│    classification, conversation, data-augmentation, domain-qa,               │
│    dynamic-few-shot, extraction, instruction-following,                      │
│    question-answer, tool-use-synthetic, tool-use                             │
│                                                                              │
│  templates/training/       9 methods + 9 model presets                       │
│    dpo, full-sft, gepa, gkd, grpo, kto, lora-sft, osft, qlora-sft          │
│    models/: llama3-8b-lora, llama3-8b-qlora, qwen-1.5b-lora,               │
│             qwen-7b-sft, qwen3-0.6b-gkd, qwen3-0.6b-lora,                  │
│             qwen3-4b-dpo, qwen3-4b-grpo, qwen3-4b-lora                      │
│                                                                              │
│  templates/eval/           19 judge templates                                │
│    classification-accuracy, code-correctness, code-maintainability,          │
│    code-performance, code-quality, code-security, completeness,              │
│    equivalence, exact-match, format-compliance, groundedness,                │
│    instruction-following, llm-judge, regex-match-phone,                      │
│    regex-no-error-keywords, relevance, safety, topic-adherence,              │
│    truthfulness                                                              │
│                                                                              │
│  examples/                 6 end-to-end examples                             │
│    content-moderator, distillation, entity-extractor,                        │
│    intent-router, summarizer, ticket-classifier                              │
│    (each: synth.yaml + train.yaml + eval.yaml)                               │
│                                                                              │
│  Recipe composition: extends: base-template, dot-notation overrides          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Job Pipeline — End-to-End Data Flow

```
PHASE 1: SDG (Synthetic Data Generation)
─────────────────────────────────────────

  User (Studio/CLI/API)
    │
    │  POST /api/v1/jobs/recipe
    │  {recipe: "examples/ticket-classifier/synth", overrides: {num_samples: 100}}
    v
  Amortized Server
    │  load_recipe() --> apply_overrides() --> flatten_to_config()
    │  strip_secrets() --> create_job(status=queued)
    v
  SQLite [jobs table]
    │
    v
  Worker (picks up job)
    │  _build_synth_config() --> asynth-compatible YAML
    │  Build JobSpec (image, cmd, env, config_files)
    v
  Compute Backend
    │  K8s: Secret + ConfigMap + Job
    │  SSH: podman run --gpus all
    │  Local: subprocess.Popen
    v
  asynth Container
    │  asynth synthesize --config synth_config.yaml
    │  Calls teacher LLM (via Gateway) to generate training data
    │  mlflow.log_artifact(generated_data.jsonl)
    v
  MLflow --> S3 (artifacts stored)
    │
    v
  Worker (completion)
    │  Extract mlflow_run_id from logs
    │  Tag run, update job status = succeeded
    v
  Job complete. mlflow_run_id stored on job record.


PHASE 2: Training (chained via parent_job_id)
──────────────────────────────────────────────

  User
    │  POST /api/v1/jobs {type: "training", parent_job_id: "<sdg_job_id>"}
    v
  Worker
    │  _resolve_parent_artifacts()
    │    --> lookup parent job --> get mlflow_run_id
    │    --> GET MLflow /runs/get --> artifact URI
    │    --> inject data_path = s3://bucket/.../generated_data.jsonl
    │  _training_hub_config_yaml() --> tool-native YAML
    │  Build JobSpec with S3Download for training data
    v
  Compute Backend
    │  K8s init container: aws s3 cp s3://...  /amortized/work/
    │  Main container: thub lora-sft --config /amortized/config.yaml
    │  TRL auto-logs to MLflow (report_to: mlflow)
    v
  Worker (completion)
    │  Extract mlflow_run_id
    │  Register model in MLflow Model Registry
    │  Update job status = succeeded


PHASE 3: Eval (chained via parent_job_id)
─────────────────────────────────────────

  User
    │  POST /api/v1/jobs {type: "eval", parent_job_id: "<training_job_id>"}
    v
  Worker
    │  Resolve parent training artifacts
    │  _resolve_judge_template() from templates/eval/
    │  _eval_config_yaml() --> asynth judge YAML
    v
  asynth Container
    │  asynth judge --config config.yaml --data eval_data.jsonl
    │  LLM judge evaluates model outputs
    │  Results logged to MLflow
    v
  Job complete. Eval scores available in MLflow.
```

---

## Kubernetes Deployment Topology

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                                          │
│                                                                              │
│  namespace: amortized-<user> (per-developer)                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │  amortized-     │  │  studio        │  │  opencode      │                 │
│  │  server         │  │  (nginx+SPA)   │  │  (morty)       │                 │
│  │  Deployment     │  │  Deployment    │  │  Deployment    │                 │
│  │  :8000          │  │  :8080         │  │  :8001         │                 │
│  │  + PVC (SQLite) │  │                │  │                │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │  claude-code    │  │  ServiceAccount │  │  ConfigMap     │                 │
│  │  Deployment     │  │  + ClusterRole  │  │  (env vars)    │                 │
│  │                 │  │  (RBAC for job  │  │                │                 │
│  │                 │  │   management)   │  │                │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  namespace: amortized-<user>-jobs (per-developer)                            │
│  ┌───────────────────────────────────────────────────────────────┐           │
│  │  Per-job resources (created dynamically by worker):           │           │
│  │                                                               │           │
│  │  K8s Secret ──> ConfigMap ──> Job                             │           │
│  │  (env vars)     (YAML config)  (pod with GPU, init container) │           │
│  │                                                               │           │
│  │  ownerReferences: Secret + ConfigMap owned by Job (auto-GC)   │           │
│  │  GPU quota: 1 GPU per developer                               │           │
│  │  RBAC: per-user roles for job management                      │           │
│  └───────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  namespace: amortized (shared services)                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │  MinIO          │  │  MLflow         │  │  Docling-Serve  │                 │
│  │  (S3-compat)    │  │  Tracking       │  │  (optional)    │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

```
Application config (pydantic-settings, AMORTIZED_ prefix):

  AMORTIZED_HOST / PORT           Server binding (0.0.0.0:8000)
  AMORTIZED_DB_PATH               SQLite path (./data/amortized.db)
  AMORTIZED_DATA_DIR              Data directory (./data)
  AMORTIZED_RECIPES_DIR           Override recipes directory (optional)
  AMORTIZED_COMPUTE_BACKEND       Default: local | ssh | kubernetes
  AMORTIZED_COMPUTE_NAMESPACE     K8s namespace (amortized-jobs)
  AMORTIZED_IMAGE_REGISTRY        Container registry (ghcr.io/amortized-ai)
  AMORTIZED_IMAGE_PULL_POLICY     Always (avoid stale cached images)
  AMORTIZED_MLFLOW_TRACKING_URI   MLflow server URL
  AMORTIZED_GATEWAY_URL           MLflow AI Gateway URL
  AMORTIZED_DOCLING_URL           Docling-serve URL (empty = disabled)
  AMORTIZED_STORAGE_BUCKET        S3 bucket name
  AMORTIZED_EXTERNAL_URL          Externally reachable server URL
  AMORTIZED_API_KEY               Bearer token (empty = no auth)
  AMORTIZED_CORS_ORIGINS          Comma-separated allowed origins
  AMORTIZED_FORWARD_ENV           Env vars forwarded to job containers
  AMORTIZED_DEFAULT_BACKEND       Default backend override (optional)

User config (~/.amortized/config.yaml):

  SSH/K8s backends, default_backend, forward_env, gateway_url, docling_url
  Created via `amortized config` interactive CLI
```

---

## CLI

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  amortized CLI — Typer app, thin REST client over httpx                      │
│                                                                              │
│  Server discovery: AMORTIZED_API_URL env var                                 │
│                    --> ~/.amortized/config.yaml (api_url)                     │
│                    --> fallback: http://localhost:8000                         │
│                                                                              │
│  Commands:                                                                   │
│    amortized up              Start API server (uvicorn)                      │
│    amortized config          Interactive backend configuration               │
│    amortized submit          Submit a job (--recipe, --model, --data,        │
│                               --set KEY=VALUE, --dry-run, --confirm)         │
│    amortized jobs            List jobs (--status, --type filters)            │
│    amortized job <id>        Get job details                                │
│    amortized logs <id>       Stream job logs (--follow)                      │
│    amortized cancel <id>     Cancel a running/queued job                     │
│    amortized types           List available job types                        │
│    amortized recipes         List available recipes                          │
│    amortized recipe <name>   Show recipe details                            │
│    amortized artifacts       List artifacts                                  │
│    amortized backends        List compute backends                          │
│    amortized upload          Upload a local file as artifact                 │
│    amortized mcp             Start MCP server (HTTP transport)              │
│    amortized health          Check API server health                        │
│                                                                              │
│  The `config` command has interactive flows for SSH backend setup            │
│  (connectivity test, GPU detection, container runtime detection)             │
│  and K8s backend setup (context discovery from ~/.kube/config).              │
│  Auto-detects API key env vars for credential forwarding.                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```
