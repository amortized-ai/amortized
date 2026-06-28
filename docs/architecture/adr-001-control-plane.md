# ADR-001: Control Plane Architecture for Task Model Creation

**Status**: Accepted
**Date**: 2026-06-28
**Authors**: Shiv

## Context

Amortized is a control plane for building task models — small, fine-tuned LLMs that replace expensive frontier model API calls for specific tasks (classification, extraction, routing, summarization). It is deployed on Red Hat OpenShift AI (RHOAI).

The platform orchestrates three sequential job types: synthetic data generation (SDG), training (LoRA SFT), and evaluation. Serving is handled by Red Hat's Model-as-a-Service (MaaS).

We evaluated reference architectures from:
- **eval-hub** (Red Hat) — evaluation orchestration on OpenShift AI with K8s Jobs, sidecar pattern, TrustyAI operator
- **granite.build** (IBM) — LLM pipeline orchestration with DAG-based builds, URI-based asset stores, OpenLineage
- **Kubeflow Pipelines** — workflow orchestration with pipeline runs
- **MLflow** — experiment tracking, model registry, artifact storage, AI gateway

The core design principle: **use RHOAI-native services where they exist, build only what's missing.**

---

## Decisions

### AD-1: MLflow is the single artifact and metadata store

**Context**: We need to track datasets (SDG outputs), models (training outputs), evaluation results, experiment metrics, and lineage between them. Options evaluated: Kubeflow Model Registry (models only, metadata pointers, alpha-stage), granite.build's ArtifactRegistration (custom build), MLflow (experiments, datasets, models, eval results, lineage, artifact storage).

**Decision**: MLflow is the single source of truth for all artifacts and metadata. No custom artifact tables in amortized.

**Consequences**:
- (+) One service handles experiment tracking, dataset tracking, model registry, artifact storage, and lineage
- (+) MLflow is already deployed on RHOAI and well-understood
- (+) MLflow 3.7+ defaults to SQLite for standalone, PostgreSQL for production — no separate DB needed
- (+) `mlflow.data` module tracks datasets as first-class entities with lineage
- (+) Model Registry provides versioning, aliases (`@champion`), and deployment URI resolution
- (-) Dataset listing requires querying runs filtered by tags — no native `list_datasets()` API. Accepted as YAGNI for v1; add a cache/index table if query performance becomes a bottleneck
- (-) MLflow Runs don't exist until the container starts — can't represent "queued" state (solved by the job table, AD-2)

### AD-2: Thin job table for operations, K8s Jobs for runtime, MLflow for science

**Context**: We need to track job submissions, status, and history. Options evaluated: pure K8s + MLflow (no custom table) vs. thin custom table as the glue layer. Every production ML platform studied (eval-hub, granite.build, Kubeflow Pipelines, Argo Workflows) maintains its own job table.

**Decision**: Amortized maintains a single `jobs` table in SQLite (dev) / PostgreSQL (prod). This is the **only table** in amortized's database.

**Schema**: `id`, `type`, `status`, `config`, `recipe`, `user_id`, `k8s_job_name`, `k8s_namespace`, `mlflow_run_id`, `mlflow_experiment`, `parent_job_id`, `error`, `created_at`, `started_at`, `completed_at`.

**Three-layer separation**:
- Job table = operations ("what was submitted, what's queued")
- K8s Jobs = runtime ("what's running, pod status, logs")
- MLflow Runs = science ("what happened — params, metrics, artifacts, lineage")

**Consequences**:
- (+) Durable record at submission time — survives server crashes
- (+) Fast queries for Studio UI (one indexed table vs. cross-service joins)
- (+) Clean separation of concerns — each system owns one responsibility
- (+) Only one table to maintain — conversations, evaluators, artifacts, api_keys tables all eliminated
- (-) Requires syncing status between K8s Job state and the table (via polling loop)

### AD-3: No Kueue for MVP, K8s Jobs dispatched directly

**Context**: Kueue (GA on OpenShift, Red Hat build 1.3) provides job queuing, GPU quota management, and fair sharing. However, it's an optional component that platform engineers must install and configure.

**Decision**: MVP dispatches K8s Jobs directly without Kueue. Kueue is a documented future enhancement.

**Consequences**:
- (+) No dependency on Kueue being installed — works on any OpenShift cluster
- (+) Simpler deployment and testing
- (-) No GPU quota management — concurrent jobs compete for GPUs via K8s scheduler best-effort
- (-) No fair sharing between users — first-come-first-served
- **Future**: Add Kueue label to K8s Jobs when Kueue is detected on the cluster

### AD-4: Polling for job status (not K8s Watch)

**Context**: K8s Watch API (informers) provides instant status updates but requires reconnection handling, periodic resync, and more complex error recovery. The current polling loop (5-second interval) works reliably.

**Decision**: MVP keeps the polling loop for job status updates. K8s Watch is a future optimization.

**Consequences**:
- (+) Simple, proven, reliable — already tested on ROSA cluster
- (+) No reconnection logic, no resync, no watch timeout handling
- (-) 0-5 second status update latency (invisible to users chatting with Morty)
- (-) Wasted API calls when no jobs are running
- **Future**: Replace with K8s Watch for instant updates and zero-waste polling

### AD-5: MLflow AI Gateway for LLM provider API keys (with fallback)

**Context**: SDG and eval jobs need LLM provider API keys (OpenAI, Anthropic, etc.). Options: custom encrypted key store in amortized's DB (current), MLflow AI Gateway (centralized, encrypted, provider-agnostic proxy).

**Decision**: Target architecture is MLflow AI Gateway. Fallback to per-job K8s Secrets if Gateway integration doesn't work with asynth/LiteLLM.

**Consequences**:
- (+) No custom key storage code — MLflow handles encryption and management
- (+) Containers never see API keys — call the gateway instead of providers directly
- (+) Provider switching without changing job configs
- (+) Built-in failover, budget tracking, traffic splitting
- (-) Requires spike test: asynth → LiteLLM → MLflow Gateway compatibility
- (-) If spike fails, fallback to per-job K8s Secrets (current approach works)
- **Action**: Spike test before committing — configure asynth with `api_base` pointing to MLflow Gateway and run an SDG job

### AD-6: RHOAI data connections for S3, no custom data connectors

**Context**: Job containers need S3 credentials, HuggingFace model downloads, and document ingestion.

**Decision**: Use RHOAI data connections (K8s Secrets with `opendatahub.io` labels) for S3. HuggingFace and Docling are pass-through (handled by the container images). Amortized builds zero custom data connectors.

**Consequences**:
- (+) Platform engineer configures S3 once via RHOAI dashboard — amortized just references the Secret
- (+) No custom credential management code
- (+) Standard RHOAI pattern — familiar to platform engineers
- (-) Depends on platform engineer setting up the data connection correctly

### AD-7: Studio is the single user-facing frontend

**Context**: MLflow has its own web UI. Running two UIs creates a fragmented experience where users bounce between Studio and MLflow.

**Decision**: Studio is the only interface data scientists use. Studio calls MLflow's REST APIs for datasets, models, metrics, and AI gateway management. MLflow UI remains accessible for platform engineers and power users.

**Consequence**: Studio and backend are developed in parallel and shipped together as a complete package. No half-baked releases.

**Studio page → backend mapping**:
- Chat → OpenCode (Morty)
- Jobs → amortized API (job table)
- Datasets → MLflow API (`runs/search` filtered by SDG tags)
- Models → MLflow API (Model Registry)
- Settings → LLM Providers → MLflow API (AI Gateway routes)
- Recipes → amortized API

**Nginx routing**: `/api/` → amortized, `/mlflow/` → MLflow, `/agent/` → OpenCode

### AD-8: Two MCP servers for Morty (amortized + MLflow)

**Context**: Morty needs access to both amortized operations (jobs, recipes) and MLflow capabilities (experiments, models, datasets, gateway). Options: single facade wrapping MLflow (more code, more maintenance) or two MCP servers (each service exposes its native capabilities).

**Decision**: Morty connects to two MCP servers — amortized (~10 tools) and MLflow (~45 tools). Total ~55 tools. Reduce only if the model struggles with tool selection accuracy.

**Consequences**:
- (+) No wrapper code — each service exposes capabilities natively
- (+) MLflow MCP tools auto-update when MLflow is upgraded
- (-) 55 tools is high — LLM tool selection may degrade. Monitor and reduce if needed
- (-) MLflow MCP requires Python in the OpenCode container — needs custom image or sidecar
- **Future**: If tool overload is a problem, use OpenCode permissions to hide irrelevant MLflow tools

### AD-9: OpenCode as agent runtime with switchable LLM

**Context**: The AI agent (Morty) guides users through task model creation. Options evaluated: custom agent loop, Claude Agent SDK, OpenCode. OpenCode is already working with Morty identity, MCP integration, and Vertex AI.

**Decision**: OpenCode is the agent runtime. LLM backend is configurable at deploy time via `.opencode.json` (no code change to switch providers). Workflow enforcement via skills (to be defined). Confirmation gates before job submission (to be defined).

**Consequences**:
- (+) Already working — tested end-to-end on ROSA cluster
- (+) 75+ LLM providers supported natively — switch by changing one config value
- (+) Built-in session management, HTTP API for Studio integration
- (-) OpenCode's default identity leaks through — requires custom agent definition with permission scoping
- (-) Workflow enforcement via system prompt is soft — skills and hooks needed for hard gates
- **Future**: Define workflow steps as OpenCode skills, implement Oumi-style confirmation cards in Studio

### AD-10: OAuth proxy sidecar for auth, no custom auth code

**Context**: Users need to authenticate to access Studio and amortized. RHOAI uses OpenShift OAuth with an `oauth-proxy` sidecar pattern.

**Decision**: Deploy an OAuth proxy sidecar in front of Studio. Amortized reads `X-Forwarded-User` header for user identity. No custom auth middleware (TokenReview, SubjectAccessReview) in amortized for MVP.

**Consequences**:
- (+) Zero auth code in amortized — OAuth proxy handles everything
- (+) Standard RHOAI pattern — well-documented, battle-tested
- (+) User identity available via header for job ownership tagging
- (-) No fine-grained authorization (all authenticated users can do everything)
- (-) No multi-tenancy beyond user-level job filtering
- **Future**: Add TokenReview + SubjectAccessReview for RBAC, namespace-per-tenant isolation

### AD-11: No serving — handoff to Red Hat MaaS

**Context**: After training and evaluation, the model needs to be deployed for inference. RHOAI provides KServe and llm-d for model serving.

**Decision**: Amortized does not manage serving. The workflow ends at model registration in MLflow Model Registry. For MVP, Morty provides MaaS deployment instructions. For v2, add a thin "deploy to MaaS" action (single K8s API call to create InferenceService).

**Consequences**:
- (+) No serving infrastructure to build or maintain
- (+) Leverages RHOAI's production-grade serving (KServe, llm-d)
- (-) UX cliff: guided workflow → manual deployment. Mitigated by Morty's handoff instructions
- **Future**: Thin deploy action — one endpoint that creates InferenceService CRD pointing to MLflow Model Registry URI

---

## Summary: What Amortized Builds vs Uses

| Concern | Amortized Builds | External Service |
|---|---|---|
| **Job management** | Job table (1 table), REST API (~10 endpoints), K8s Job dispatch, polling loop | K8s for runtime, Kueue for queuing (future) |
| **Artifacts** | MLflow env var injection, run ID extraction | MLflow (experiments, datasets, models, registry, lineage) |
| **Data connectors** | Nothing | RHOAI data connections (S3), MLflow AI Gateway (LLM keys) |
| **Agent** | Morty agent definition, skills (future), system prompt | OpenCode (runtime), Vertex AI / Claude (LLM) |
| **Frontend** | Studio (React SPA) | MLflow API (backend for datasets/models/metrics) |
| **Auth** | `X-Forwarded-User` reading | OAuth proxy sidecar |
| **Serving** | Handoff instructions (v1), thin deploy action (v2) | KServe / llm-d / MaaS |
| **MCP** | fastapi-mcp (auto-generated from OpenAPI) | MLflow MCP (45 tools) |

## MVP Scope

The MVP includes:
- Amortized server with job table (1 table), ~10 REST API endpoints, K8s Job dispatch with polling
- MLflow for all artifact/experiment/model tracking
- Studio with Chat (Morty), Jobs, Datasets (via MLflow), Models (via MLflow), Recipes, Settings
- OpenCode (Morty) with two MCP servers (amortized + MLflow)
- OAuth proxy for auth
- SDG, Training, Eval job types (workload details to be designed separately)

The MVP does NOT include:
- Kueue integration
- K8s Watch (uses polling)
- Serving/deployment
- Multi-tenancy (namespace isolation)
- Custom auth middleware
- Workflow enforcement skills (uses system prompt)
