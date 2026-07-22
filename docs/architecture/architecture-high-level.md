# Amortized — High-Level Architecture

```
                              ┌─────────────────────────────────────┐
                              │            ENTRY POINTS             │
                              │                                     │
                              │   Browser    CLI    External Agent   │
                              │      │        │          │          │
                              └──────┼────────┼──────────┼──────────┘
                                     │        │          │
                                     v        │          │
┌────────────────────────────────────────────────────────────────────────────────┐
│                          AMORTIZED  PLATFORM                                   │
│                                                                                │
│  ┌──────────────────────┐     ┌──────────────────────────────────────────────┐ │
│  │   Studio (React SPA) │     │            Nginx Reverse Proxy               │ │
│  │                      │     │                                              │ │
│  │  - Job Dashboard     │────>│  /          -> Static files (SPA)            │ │
│  │  - Recipe Browser    │     │  /api/      -> Amortized Server (:8000)      │ │
│  │  - Cost Calculator   │     │  /agent/    -> Agent Service (:8001)         │ │
│  │  - Chat (Morty)      │     │  /mlflow/   -> MLflow (:5000)               │ │
│  │                      │     │  /mcp       -> MCP Server (SSE)             │ │
│  └──────────────────────┘     └───────┬──────────┬──────────┬───────────────┘ │
│                                       │          │          │                  │
│          ┌────────────────────────────┘          │          └──────────┐       │
│          v                                       v                    v       │
│  ┌───────────────────────────────┐   ┌───────────────────────┐               │
│  │   Amortized Server (FastAPI)  │   │  Agent Service (Morty) │               │
│  │                               │   │                       │               │
│  │  REST API  (/api/v1/*)        │<──│  Claude Agent SDK     │               │
│  │  MCP Server (/mcp)       <────│───│  MCP: amortized + mlflow              │
│  │  Background Worker            │   │  Prompts: soul + workflows            │
│  │  Config Translator            │   └───────────────────────┘               │
│  │  SQLite (jobs)                │          │          │                      │
│  └───────────┬───────────────────┘          │          │──> CLI ──────────────┘
│              │                              │
│              │  Dispatches jobs              │  Calls teacher LLMs
│              v                              v
│  ┌──────────────────────────────────────────────────────┐
│  │               COMPUTE BACKENDS                       │
│  │                                                      │
│  │   Kubernetes          SSH               Local        │
│  │   (K8s Jobs)      (Remote GPU)      (Subprocess)     │
│  │                                                      │
│  └──────────┬──────────────┬──────────────┬─────────────┘
│             │              │              │
└─────────────┼──────────────┼──────────────┼──────────────────────────────────┘
              │              │              │
              v              v              v
┌──────────────────────────────────────────────────────┐
│                ML TOOL CONTAINERS                     │
│                                                      │
│   asynth (SDG + Eval)        training-hub / TRL      │
│   - asynth synthesize        - thub <algo> --config   │
│   - asynth judge             - trl <algo> --config    │
│                                                      │
└──────────┬───────────────────────────┬───────────────┘
           │                           │
           v                           v
┌──────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE SERVICES                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │    MLflow     │  │  S3 / MinIO  │  │   MLflow AI Gateway    │ │
│  │              │  │              │  │                        │ │
│  │  Tracking    │  │  Artifacts   │  │  Routes to teacher     │ │
│  │  Experiments │──│  Models      │  │  LLMs:                 │ │
│  │  Model       │  │  Datasets    │  │  - OpenAI (GPT-4o)     │ │
│  │  Registry    │  │  Documents   │  │  - Anthropic (Claude)   │ │
│  │              │  │              │  │  - Google (Vertex AI)   │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Docling-Serve (optional)                                    │ │
│  │                                                              │ │
│  │  Document processing: PDF, DOCX, HTML --> structured text    │ │
│  │  Amortized proxies requests, stores results in MLflow        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
Job Pipeline:
  User Request
    --> Studio (or CLI / MCP)
      --> Nginx
        --> Amortized Server (FastAPI)
          --> SQLite (persist job, status=queued)
          --> Worker picks up job (polls every 2s)
            --> Translate config to tool-native YAML
            --> Resolve parent job artifacts (MLflow)
            --> Dispatch to compute backend
              --> K8s Job / SSH container / Local subprocess
                --> ML container runs (asynth or training-hub)
                  --> Logs metrics + artifacts to MLflow --> S3
          --> Worker polls completion, extracts MLflow run ID
          --> Registers trained model in MLflow Model Registry

Document Processing:
  User Upload / URL
    --> Amortized Server
      --> Proxy to Docling-Serve (convert file/URL)
      --> Store parsed content + source in MLflow ("amortized/documents")
      --> Return structured content to user
```
