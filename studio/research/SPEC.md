# Amortized Studio — Product Specification

## Problem Statement

AI teams building production applications use frontier models (GPT-4o, Claude, etc.) for tasks that don't need frontier-level intelligence: classification, extraction, routing, summarization. These tasks are specific, repeatable, and learnable. A small fine-tuned "task model" can do them faster, cheaper, and more reliably — but the process of building one (generating training data, fine-tuning, evaluating, deploying) requires stitching together multiple tools, writing YAML configs, and running CLI commands across local and remote machines.

The [amortized](https://github.com/amortized-ai/amortized) library already provides the backend runtime for this pipeline — synthetic data generation via asynth, training via TRL, evaluation via LLM judges, serving via vLLM, job orchestration, compute backends (local + SSH), and an agent chat system. But there is no visual interface. Users interact through CLI commands or raw API calls.

**Studio is the web UI that makes the amortized runtime accessible to teams who want to build task models without living in the terminal.**

## Solution

A web application (Amortized Studio) that provides a chat-first, visually rich interface for building task models on your own infrastructure. The chat agent guides users through the full pipeline — from describing a task in natural language to having a deployed model — while dashboard views provide monitoring, data inspection, and configuration for power users.

The Studio connects to the existing amortized FastAPI server and its REST API / MCP server. It does not replace the backend — it is a client.

### Design Principles

1. **Chat-first, dashboard-second.** The agent chat is the primary entry point. Users describe what they want; the agent configures and submits jobs. Dashboard pages exist for monitoring and inspection, not as the primary workflow.

2. **One step at a time.** The agent walks users through the pipeline sequentially. It never dumps a full plan or asks more than 1-2 questions at once. Each step builds on the previous one.

3. **Structured interactions over free text.** When the agent asks a question, it presents clickable option cards with descriptions — not open-ended prompts. Users can always type freely, but the cards reduce cognitive load.

4. **Show the data.** After every generation or training step, show sample data, metrics, and quality indicators before proceeding. Trust is built through transparency.

5. **Self-hosted, own your data.** Everything runs on the user's infrastructure. No cloud dependency. The Studio is a static frontend that talks to the user's amortized server.

## User Stories

1. As a team lead, I want to open a web dashboard and see all my projects at a glance, so that I can quickly navigate to the task model I'm working on.

2. As a developer, I want to describe a task in natural language (e.g., "build a ticket classifier"), so that the agent can guide me through generating data, training, and evaluating without me writing YAML configs.

3. As a developer, I want the chat agent to present clickable option cards (e.g., "Customer support", "IT helpdesk", "Something else") instead of asking open-ended questions, so that I can make decisions faster.

4. As a developer, I want to see a plan progress indicator (e.g., "Step 2/6: Define evaluators") during the agent-guided flow, so that I know where I am in the pipeline and how much is left.

5. As a developer, I want to see "tool result" badges inline in the chat when the agent calls backend APIs, so that I understand what the agent is doing on my behalf.

6. As a developer, I want the agent to propose actions (e.g., "Submit training job") with a confirm button rather than executing them silently, so that I maintain control.

7. As a developer, I want to upload source documents (PDFs, JSONL, CSV, TXT) via drag-and-drop, so that they can be used as input for synthetic data generation.

8. As a developer, I want to browse my datasets in a table showing sample rows (system/user/assistant messages), so that I can verify the data quality before training.

9. As a developer, I want automatic quality tests to run on every dataset (token limits, alternating turns, empty turns, missing user messages, system message positioning), so that I catch data issues before they waste GPU time.

10. As a developer, I want to see a quality test summary badge (e.g., "5/5 Tests Passed") on the dataset list, so that I can quickly identify problematic datasets.

11. As a developer, I want dataset versioning, so that I can track how my training data evolves over time.

12. As a developer, I want to see dataset lineage (e.g., "Used in 2 Training jobs"), so that I can trace which models were trained on which data.

13. As a developer, I want to browse and manage evaluators (LLM judges) with their prompt templates, judgment types, and model configurations, so that I can customize how my models are assessed.

14. As a developer, I want to run evaluations against a dataset using selected evaluators, so that I can measure model quality on specific criteria.

15. As a developer, I want to compare evaluation results across multiple models side-by-side, so that I can determine whether fine-tuning improved quality.

16. As a developer, I want to see training loss curves and gradient norm charts on the model detail page, so that I can assess training convergence without needing W&B.

17. As a developer, I want the model detail page to show full lineage (base model → training dataset → training recipe), so that I can reproduce the training run.

18. As a developer, I want to see a list of all jobs (SDG, training, eval, serve) with status badges, progress bars, and duration, so that I can monitor what's running.

19. As a developer, I want to click a job row to see a detail panel with logs, metrics, config JSON, and error messages, so that I can debug failures.

20. As a developer, I want to filter jobs by type (training/sdg/eval/serve) and status (running/succeeded/failed), so that I can find specific jobs quickly.

21. As a developer, I want to cancel a running job from the UI, so that I don't have to use the CLI.

22. As a developer, I want to configure training jobs through a form-based recipe builder with collapsible sections (Training Method, Model Selection, Data, Training Settings), so that I don't have to write YAML manually.

23. As a power user, I want a split-pane JSON editor (schema outline + raw JSON) for recipe configuration, so that I can see and edit the full config when needed.

24. As a developer, I want to save recipe configurations and reuse them, so that I don't reconfigure from scratch each time.

25. As a developer, I want an "Execute" button on the recipe builder that submits the job directly, so that I can go from config to running in one click.

26. As a developer, I want to configure compute backends (local GPU, SSH remotes) through the UI, so that I don't have to edit config files.

27. As a developer, I want to manage API keys (OpenAI, Anthropic, etc.) through the UI with encrypted storage, so that I don't have to set environment variables.

28. As a developer, I want to deploy a trained model for serving (via vLLM) from the model detail page, so that I can test my fine-tuned model immediately.

29. As a developer, I want to see GPU utilization and availability on the compute settings page, so that I know if my machine can handle the job I'm about to submit.

30. As a team member without CLI access, I want to do everything through the web UI (the chat agent handles all technical work), so that I can contribute to model building without terminal skills.

31. As a developer, I want the agent chat to be available contextually on data/model pages (as a side panel), so that I can ask questions about the artifact I'm looking at.

32. As a developer, I want WebSocket-based real-time updates on job progress, so that I see status changes without refreshing.

33. As a developer, I want to preview artifacts (generated data, training metrics, logs) inline, so that I can assess quality without downloading files.

34. As a developer, I want to estimate GPU VRAM requirements before submitting a training job, so that I don't waste time on jobs that will OOM.

35. As a developer, I want to resume a failed training job from the last checkpoint, so that I don't lose partial progress.

## Implementation Decisions

### Architecture

- **Frontend**: Single-page application (SPA) that communicates with the amortized FastAPI backend via REST API and WebSocket.
- **No separate backend**: The Studio is a pure frontend client. All business logic, job orchestration, and data storage remain in the amortized server. The Studio makes HTTP requests to the existing API endpoints.
- **The amortized server already provides**: REST API for jobs/artifacts/datasets/evaluators/events, WebSocket for real-time job events, SSE for agent chat streaming, MCP server for external agent integration.

### Module Breakdown

#### Module 1: Chat Agent Interface

The primary interaction surface. Connects to the amortized agent chat endpoints (`/api/v1/agent/chat`).

- **Streaming responses** via SSE from the existing `stream_message()` endpoint.
- **Structured option cards**: When the agent response contains structured choices, render them as clickable cards with title + description. The agent already supports `suggested_action` in its response model — extend this to support multi-choice prompts.
- **Tool result indicators**: Show collapsible "Tool result" badges when the agent calls backend tools (the stream already emits `thinking` and `tool_result` events).
- **Proposed action cards**: The agent already uses `propose_action` to render confirm/reject buttons for job submission. Render these as prominent action cards.
- **Plan progress indicator**: A step tracker (e.g., "Step 2/6: Define evaluators") derived from the agent's conversation state.
- **Conversation history**: Stored server-side in the existing `conversations` and `messages` tables.
- **Contextual chat panel**: A collapsible side panel version of the chat available on Data/Model pages.

#### Module 2: Job Dashboard

Lists and monitors all jobs across the pipeline.

- **Data source**: `GET /api/v1/jobs` with optional `?status=` and `?type=` filters.
- **Real-time updates**: Subscribe to the WebSocket endpoint (`/api/v1/ws`) for job state change events.
- **Job detail panel**: Slide-out panel (not a full page navigation) showing metadata, config JSON, error messages, and timestamps.
- **Job actions**: Cancel (`POST /api/v1/jobs/{id}/cancel`), resume (`POST /api/v1/jobs/{id}/resume`).
- **Training metrics**: For training jobs, fetch metrics via `GET /api/v1/jobs/{id}/metrics` and render loss curves + gradient norm charts.

#### Module 3: Dataset Manager

Browse, inspect, and manage training datasets and source files.

- **Dataset list**: `GET /api/v1/artifacts?type=dataset` — table with name, version, created date, quality test badge, row count.
- **Dataset detail**: Tabs for Overview (metadata + sample rows), Versions, Quality Tests.
- **Sample row preview**: Fetch first N rows via `GET /api/v1/artifacts/{id}/preview` and render as a table showing message roles and content.
- **Quality tests**: Client-side checks on the preview data (token count validation, turn alternation, empty turns, missing user messages, system message position). Run automatically on dataset load.
- **File upload**: Drag-and-drop zone for source files. `POST /api/v1/datasets/upload` for JSONL/CSV. Separate file storage for PDFs/documents used in document-grounded synthesis.
- **Dataset lineage**: Show which jobs consumed this dataset (query jobs by artifact reference).

#### Module 4: Evaluator Registry

CRUD interface for evaluation judges.

- **Evaluator list**: `GET /api/v1/evaluators` — table with name, type (LLM/rule-based), judgment type, model.
- **Evaluator detail/edit**: Full view of the evaluator's prompt template, variables, model configuration, inference parameters.
- **Create evaluator**: Form with fields for name, description, type, prompt template, judgment type, response format, variables, model, inference params.
- **Default evaluators**: The amortized server seeds default evaluators on startup — display these as system evaluators that can be cloned but not deleted.

#### Module 5: Evaluation Runner & Results

Run evaluations and compare results.

- **Run evaluation**: Select evaluator(s) + dataset + model → submit eval job.
- **Evaluation results**: Table with per-sample pass/fail, scores, explanations. Summary stats (pass rate, average score).
- **Model comparison**: Side-by-side view of eval results across multiple models on the same dataset. Highlight where the fine-tuned model beats or underperforms the baseline.

#### Module 6: Model Registry

Browse trained models with full observability.

- **Model list**: `GET /api/v1/artifacts?type=adapter_weights` — table with name, type (TUNED), base model, dataset, created date.
- **Model detail**: Two-column layout. Left: Model Info (ID, version, learning rate, dataset size). Right: Model Reference (base model, training dataset link, training recipe link). Below: Loss Curve chart, Gradient Norm chart (rendered from training metrics data).
- **Model actions**: Evaluate (link to eval runner), Export (download adapter weights), Deploy (submit serve job via vLLM).

#### Module 7: Recipe Builder

Form-based configuration for all job types.

- **Recipe list**: `GET /api/v1/recipes` — table with name, type (Training/SDG/Eval), description, version.
- **Builder form**: Collapsible sections per job type:
  - **Training**: Training Method (SFT/DPO/GRPO/KTO), Model Selection (base model dropdown), Data (training + validation dataset pickers), Training Settings (learning rate, epochs, batch size, LoRA config — collapsed by default).
  - **SDG**: Teacher model, num_samples, strategy params, input data/documents.
  - **Eval**: Dataset, evaluators, judge model, metrics.
- **JSON editor**: Split-pane modal — schema outline with field descriptions (left) + syntax-highlighted JSON editor (right). Changes apply on Save.
- **Execute**: Submit the job directly from the builder with a confirmation step.
- **Save/Save As**: Persist recipe configurations for reuse.

#### Module 8: Compute & Settings

System configuration.

- **Compute backends**: List registered backends (local, SSH). Add new SSH backends with host/user/key configuration. Health check status.
- **API key management**: Add/remove provider API keys (stored encrypted via `POST /api/v1/settings/api-keys`). Key preview (last 4 chars only).
- **GPU info**: Display GPU availability from the health endpoint (`GET /api/v1/health`).
- **VRAM estimator**: Form to estimate GPU VRAM requirements before training (`POST /api/v1/estimate`).

### Key Technical Decisions

- **Agent chat protocol**: The amortized agent already streams via SSE with event types `thinking`, `tool_result`, `delta`, `action`, `done`, `error`. The frontend consumes these directly. Structured option cards will be encoded as a new field in the `AgentResponse` model (or derived from the agent's markdown output with a convention like rendering bullet-pointed options as cards).

- **Real-time job updates**: The amortized server already has a WebSocket endpoint at `/api/v1/ws` that broadcasts job events. The frontend subscribes on mount and updates job status/progress in real-time.

- **Charts**: Training loss curves and gradient norm charts rendered client-side from the `GET /api/v1/jobs/{id}/metrics` endpoint which returns `TrainingMetric` objects (step, loss, learning_rate, grad_norm).

- **Quality tests are client-side**: The dataset preview data is fetched via `GET /api/v1/artifacts/{id}/preview`, and quality checks (token counting, turn validation) run in the browser. This avoids adding new backend endpoints.

- **No auth in v1**: The amortized server has optional API key auth (`Authorization: Bearer <key>`). The Studio will support passing this key via a settings page or environment variable. No user accounts or multi-tenancy in v1.

- **Database**: All state lives in the amortized server's SQLite database. The Studio is stateless — browser state only for UI preferences (sidebar collapsed, theme, etc.).

## Testing Decisions

Good tests for the Studio verify external behavior that a user would observe, not internal implementation details. Since this is a frontend application backed by an existing API, the most valuable tests are:

### Integration tests (API contract verification)

- Verify the Studio correctly calls amortized API endpoints with the right parameters.
- Mock the amortized server responses and verify the UI renders the expected state.
- Test the WebSocket subscription lifecycle (connect, receive events, reconnect on disconnect).
- Test SSE streaming for agent chat (partial responses, tool results, proposed actions).

### Component tests (key interactive modules)

- **Chat Agent**: Test that structured option cards render from agent responses, that clicking a card sends the right message, that the plan progress indicator advances correctly, that tool result badges appear and are collapsible.
- **Recipe Builder**: Test form ↔ JSON synchronization (editing the form updates the JSON, editing the JSON updates the form). Test collapsible sections. Test dataset picker selection/deselection.
- **Quality Tests**: Test each quality check against known good and bad dataset samples (token overflow, non-alternating turns, empty messages, etc.).
- **Job Dashboard**: Test status badge rendering for all states (queued, provisioning, running, succeeded, failed, cancelled). Test progress bar calculations. Test filter interactions.
- **Charts**: Test that loss curve and gradient norm charts render correctly from training metrics data.

### E2E tests (critical user flows)

- Full chat-guided flow: describe a task → agent asks clarifying questions → user selects options → agent proposes SDG job → user confirms → job appears in dashboard.
- Recipe builder flow: open recipe → modify settings → switch to JSON view → verify changes → execute → job created.
- Dataset inspection flow: upload file → dataset appears in list → click to see detail → quality tests pass → sample rows visible.

## Out of Scope

- **User accounts, authentication, multi-tenancy**: The Studio is a single-user/single-team tool in v1. Auth is handled by the optional API key on the amortized server.
- **Mobile responsiveness**: Desktop-first. The Studio is a workstation tool.
- **Customizable dashboards or widgets**: Fixed layout in v1. No drag-and-drop dashboard customization.
- **Built-in model inference/playground**: Users cannot chat with their fine-tuned models in the Studio. They deploy via vLLM and test externally. (Future: Model Arena for side-by-side comparison.)
- **Git integration or version control**: No git-native workflow for configs or datasets. Recipes are stored in the amortized database.
- **Team collaboration features**: No commenting, ratings, or approval workflows. Single-user experience.
- **Billing or usage tracking**: No cost estimation beyond VRAM estimates. No metering.
- **Custom themes or white-labeling**: Single design system.

## Further Notes

### Competitive Context

The closest competitors are:

- **Oumi Platform** (commercial) — The most polished competitor. Chat-first with structured option cards, plan progress indicator, automatic quality tests, built-in loss curves. Cloud-hosted training. Our key differentiator: self-hosted on your own GPUs.
- **Kiln AI** (open source) — Desktop app with task-centric data model, SDG UI with topic trees, team collaboration. Outsources training to Fireworks/Together. Our differentiator: we run TRL locally.
- **Unsloth Studio** (open source) — Local web UI with Data Recipes node graph, Model Arena, training observability. Training-only, no SDG pipeline. Our differentiator: full SDG → Train → Eval → Serve loop.

### What to prioritize from competitive research

1. **Structured option cards in chat** (from Oumi) — the single highest-impact UX improvement over plain text chat.
2. **Automatic quality tests on datasets** (from Oumi) — cheap to implement, prevents wasted GPU time.
3. **Built-in loss curves on model detail** (from Oumi/Unsloth) — eliminates W&B dependency for basic training observability.
4. **Plan progress indicator in chat** (from Oumi) — shows users where they are in the multi-step flow.
5. **Slide-out job detail panels** (from Oumi) — keeps context, avoids full-page navigation.
6. **Split-pane JSON editor for recipes** (from Oumi) — bridges form-based and code-based configuration.

### Screenshots

All competitive and Oumi Platform screenshots are saved in `research/screenshots/` for reference during implementation. See the directory structure:

```
research/screenshots/
├── oumi/           # 27 screenshots across 8 sections
│   ├── overview/
│   ├── data/
│   ├── evaluators/
│   ├── models/
│   ├── deployments/
│   ├── recipes/
│   ├── jobs/
│   └── chat/
└── competitors/    # 11 screenshots across 7 products
    ├── kiln/
    ├── openpipe/
    ├── together/
    ├── unsloth/
    ├── braintrust/
    ├── fireworks/
    └── comparison/
```

### Amortized API Reference

The Studio builds on the existing amortized server API. Key endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Health check + GPU info |
| `/api/v1/jobs` | GET/POST | List/create jobs |
| `/api/v1/jobs/{id}` | GET | Job detail |
| `/api/v1/jobs/{id}/cancel` | POST | Cancel job |
| `/api/v1/jobs/{id}/resume` | POST | Resume failed job |
| `/api/v1/jobs/{id}/metrics` | GET | Training metrics |
| `/api/v1/jobs/{id}/events` | GET | Job event stream |
| `/api/v1/artifacts` | GET/POST | List/register artifacts |
| `/api/v1/artifacts/{id}` | GET | Artifact detail |
| `/api/v1/artifacts/{id}/preview` | GET | Preview artifact content |
| `/api/v1/recipes` | GET | List recipes |
| `/api/v1/recipes/{name}` | GET | Get recipe by name |
| `/api/v1/datasets/upload` | POST | Upload dataset file |
| `/api/v1/evaluators` | GET/POST | List/create evaluators |
| `/api/v1/evaluators/{id}` | GET/PUT/DELETE | Evaluator CRUD |
| `/api/v1/evaluations` | POST | Run evaluation |
| `/api/v1/judge` | POST | Judge data quality |
| `/api/v1/judge/templates` | GET | List judge templates |
| `/api/v1/estimate` | POST | Estimate GPU VRAM |
| `/api/v1/compute/backends` | GET | List compute backends |
| `/api/v1/settings/api-keys` | GET/POST/DELETE | API key management |
| `/api/v1/agent/chat` | POST | Agent chat (non-streaming) |
| `/api/v1/agent/chat/stream` | POST | Agent chat (SSE streaming) |
| `/api/v1/ws` | WebSocket | Real-time job events |
| `/mcp` | MCP | MCP server for external agents |

### Data Model Summary

The amortized server uses SQLite with these core tables:

- **jobs**: id, type (training/sdg/eval/serve), status (state machine: validating→queued→provisioning→running→succeeded/failed/cancelled), config (JSON), metadata, timestamps, output_dir, backend_handle
- **artifacts**: id, job_id, artifact_type (adapter_weights/dataset/metrics/logs), path, size, metadata, producer_job
- **events**: id, job_id, type, data (JSON), timestamp
- **evaluators**: id, name, type (llm/rule_based), prompt, judgment_type, model, inference_params
- **evaluations**: id, evaluator_id, dataset_artifact_id, job_id, status, results
- **conversations**: id, title, timestamps
- **messages**: id, conversation_id, role (user/assistant), content (JSON), timestamp
- **api_keys**: id, name, provider, key_value (encrypted), timestamp
