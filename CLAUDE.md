# Amortized — AI Model Customization Studio

## Project Purpose

Amortized is a fully open-source, on-premises studio for optimizing AI agent workflows. It replaces expensive frontier model calls with smaller, customized models — without sacrificing quality. Built on [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) (LoRA fine-tuning) and [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub) (synthetic data generation).

## Monorepo Layout

```
amortized/
├── CLAUDE.md              # This file — project conventions
├── factory.md             # Factory configuration and eval dimensions
├── server/                # Python FastAPI backend
│   ├── pyproject.toml
│   ├── src/amortized/
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI app entry point
│   │   ├── config.py      # Settings via pydantic-settings
│   │   ├── models.py      # Pydantic models
│   │   ├── worker.py      # Background job worker
│   │   ├── core/          # Domain logic (Phase 2)
│   │   ├── api/           # HTTP routers
│   │   ├── db/            # Database layer
│   │   ├── agent/         # AI agent subsystem
│   │   └── runners/       # Job execution runners
│   └── tests/
├── studio/                # Next.js React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   └── src/app/
├── .claude/skills/        # Claude Code skills for server API operations
├── docker/                # Container definitions
│   └── Dockerfile.runtime
├── eval/                  # Factory eval harness
│   └── score.py
└── .github/workflows/     # CI pipelines
    └── ci.yml
```

## Dev Commands

### Server (Python FastAPI backend)

```bash
cd server
pip install -e '.[dev]'
uvicorn amortized.main:app --reload
```

- API runs at http://localhost:8000
- Health check: GET /api/v1/health

### Studio (Next.js frontend)

```bash
cd studio
npm install
npm run dev
```

- Dev server runs at http://localhost:3000
- Proxies /api requests to the runtime backend at http://localhost:8000

## Python Conventions (server/)

- **Linter**: `ruff check src/ tests/` — enforced in CI
- **Formatter**: `ruff format src/ tests/`
- **Type checker**: `mypy src/` — strict mode enabled
- **Tests**: `pytest` — use `pytest-asyncio` for async tests
- **Package structure**: src layout (`src/amortized/`)
- **Settings**: Use `pydantic-settings` for configuration (env vars, .env files)
- **API versioning**: All endpoints under `/api/v1/`

## TypeScript Conventions (studio/)

- **Linter**: `npx eslint .`
- **Type checking**: TypeScript strict mode enabled in tsconfig.json
- **Styling**: Tailwind CSS with dark theme
- **Components**: shadcn/ui component library
- **App Router**: Next.js 15 App Router (src/app/ directory)

## Runtime API Endpoints

- POST /api/v1/jobs/training — Create a LoRA SFT training job
- POST /api/v1/jobs/sdg — Create a synthetic data generation job
- GET /api/v1/jobs — List all jobs (optional filters: status, type)
- GET /api/v1/jobs/{id} — Get job details
- GET /api/v1/jobs/{id}/metrics — Get training metrics (loss, LR, epoch per step)
- GET /api/v1/jobs/{id}/artifacts — List job output artifacts
- DELETE /api/v1/jobs/{id} — Cancel a job
- GET /api/v1/flows — List available SDG flows
- POST /api/v1/estimate — Estimate GPU VRAM requirements
- GET /api/v1/health — Health check
- POST /api/v1/agent/chat — Send a message to the AI assistant
- POST /api/v1/agent/chat/stream — Stream a response via SSE

## Training Hub — Full LoRA SFT API

Training Hub provides LoRA fine-tuning via a Python API. Install: `pip install training-hub[lora]` then `pip install training-hub[cuda]`.

### Minimal usage (3 required params)

```python
from training_hub import lora_sft

result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",  # required — HuggingFace model ID
    data_path="./data.jsonl",                    # required — training data
    ckpt_output_dir="./outputs",                 # required — output directory
)
```

### All parameters with defaults

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    # Hyperparameters
    learning_rate=2e-4,
    num_epochs=3,
    micro_batch_size=2,
    max_seq_len=2048,
    bf16=True,
    # LoRA configuration
    lora_r=16,           # LoRA rank — higher = more parameters, more expressive
    lora_alpha=32,       # LoRA alpha — scaling factor, typically 2x lora_r
    # QLoRA (4-bit quantization for reduced VRAM)
    load_in_4bit=False,  # Enable QLoRA — fits 7B+ on 24GB, 20B+ with QLoRA
)
```

### Return value

`lora_sft()` returns `{'model': model, 'tokenizer': tokenizer, 'trainer': trainer}` (live objects).

### Metrics output

Training writes `training_metrics.jsonl` to `ckpt_output_dir` with per-step records:
```json
{"step": 10, "loss": 2.345, "epoch": 1.0, "learning_rate": 1e-05, "max_steps": 1000}
```

### Checkpoints

Output is in HuggingFace PEFT format: `adapter_model.safetensors` + `adapter_config.json`, plus tokenizer files. Intermediate checkpoints in `checkpoint-N/` subdirectories.

### Memory estimation

```python
from training_hub import LoRAEstimator, QLoRAEstimator

estimator = LoRAEstimator()  # or QLoRAEstimator() for 4-bit
vram_gb = estimator.estimate(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    lora_r=16,
    batch_size=2,
    max_seq_len=2048,
)
```

### Recommended models

- **Qwen/Qwen2.5-1.5B-Instruct** — small, fast, good default for most tasks
- For 7B+ models, recommend QLoRA (`load_in_4bit=True`) to fit in VRAM
- A single 24GB GPU can fine-tune 7B with LoRA, 20B+ with QLoRA

## SDG Hub — Full Flow API

SDG Hub provides synthetic data generation. Install: `pip install sdg-hub`.

### Flow discovery and execution

```python
from sdg_hub import FlowRegistry, Flow
from datasets import Dataset

# Discover available flows
FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path("epic-jade-656")  # by flow ID

# Load and configure
flow = Flow.from_yaml(flow_path)
flow.set_model_config(
    model="openai/gpt-4o",        # teacher model (100+ providers via LiteLLM)
    api_base="http://localhost:8101/v1",  # default API base
    api_key="sk-...",
)

# Target specific blocks (optional)
flow.set_model_config(model="anthropic/claude-sonnet-4-20250514", blocks=["gen_qa_pairs"])

# Generate
result = flow.generate(
    dataset,
    runtime_params={"gen_extractive_summary": {"n": 50, "temperature": 0.7}},
    checkpoint_dir="./checkpoints",
    save_freq=50,
    log_dir="./logs",
    max_concurrency=10,
)
```

### Teacher model providers (via LiteLLM)

Supports 100+ providers: OpenAI, Anthropic, Google, vLLM (`hosted_vllm/`), Ollama (`ollama/`), Azure, Bedrock, Cohere, Mistral, Together, Groq, OpenRouter, and more.

### Dry run (validate and estimate cost)

```python
flow.dry_run(dataset, sample_size=2, enable_time_estimation=True)
```

### Progress monitoring

- **Checkpointing**: `FlowCheckpointer` saves `checkpoint_NNNN.jsonl` + `flow_metadata.json`
- **Progress query**: `FlowCheckpointer.get_progress_info()` for programmatic progress
- **Logs**: `{flow_name}_{timestamp}.log` + `{flow_name}_{timestamp}_metrics.json`
- **MLflow tracing**: Built-in via `mlflow-tracing` dependency

### Available SDG flow categories

| Category | Purpose |
|---|---|
| knowledge_infusion | Q&A generation, summaries, knowledge extraction |
| evaluation | RAG evaluation, answer quality assessment |
| agentic | MCP distillation, agent behavior datasets |
| red_team | Adversarial prompt generation |
| text_analysis | Classification, sentiment, text transformation |
| code_evaluation | Code quality, bug detection datasets |

### Block types in flows

| Category | Key Blocks | Purpose |
|---|---|---|
| llm | LLMChatBlock, PromptBuilderBlock, LLMResponseExtractorBlock | LLM calls, prompt construction, response extraction |
| parsing | TagParserBlock, RegexParserBlock, JSONParserBlock | Extract structured data from LLM output |
| transform | TextConcatBlock, RenameColumnsBlock, SamplerBlock | Data manipulation |
| filtering | ColumnValueFilterBlock | Quality filtering |
| agent | AgentBlock, MCPAgentBlock | External agent integration |
| code | PythonInterpreterBlock | Sandboxed code execution |

## The Amortization Workflow

The core workflow to replace expensive frontier model calls with smaller, customized models:

1. **Analyze** — Understand the user's agent task, identify which LLM calls are expensive and repetitive
2. **Generate data (SDG)** — Use a teacher model (e.g. GPT-4o) to generate training data via SDG Hub flows
3. **Fine-tune (LoRA SFT)** — Train a small model (e.g. Qwen 1.5B) on the generated data using Training Hub
4. **Evaluate** — Compare the fine-tuned model against the original frontier model on the task
5. **Deploy** — Replace the frontier model call with the smaller, cheaper fine-tuned model

This process "amortizes" the cost of the frontier model across many future inferences.
