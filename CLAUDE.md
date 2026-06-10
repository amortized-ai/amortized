# Amortized — AI Model Customization Studio

## Project Purpose

Amortized is a fully open-source, on-premises studio for optimizing AI agent workflows. It replaces expensive frontier model calls with smaller, customized models — without sacrificing quality. Built on [TRL](https://github.com/huggingface/trl) (fine-tuning via CLI) and [asynth](https://github.com/amortized-ai/asynth) (synthetic data generation).

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

### Git Hooks

```bash
git config core.hooksPath .githooks
```

Enables auto-regeneration of OpenAPI snapshot on commit.

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

- POST /api/v1/jobs/training — Create a training job (SFT, DPO, GRPO, KTO)
- POST /api/v1/jobs/sdg — Create a synthetic data generation job
- GET /api/v1/jobs — List all jobs (optional filters: status, type)
- GET /api/v1/jobs/{id} — Get job details
- GET /api/v1/jobs/{id}/metrics — Get training metrics (loss, LR, epoch per step)
- GET /api/v1/jobs/{id}/artifacts — List job output artifacts
- DELETE /api/v1/jobs/{id} — Cancel a job
- GET /api/v1/flows — List available SDG flows
- GET /api/v1/flows/capabilities — Report asynth capabilities and features
- POST /api/v1/judge — Judge data quality using asynth judge templates
- GET /api/v1/judge/templates — List available judge templates
- POST /api/v1/estimate — Estimate GPU VRAM requirements
- GET /api/v1/health — Health check
- POST /api/v1/agent/chat — Send a message to the AI assistant
- POST /api/v1/agent/chat/stream — Stream a response via SSE

## TRL — Training CLI

Training jobs run via the official HuggingFace TRL CLI in the `docker.io/huggingface/trl` container image.

### Supported algorithms

| Algorithm | CLI Command | Data Format |
|---|---|---|
| SFT (LoRA or full) | `trl sft` | `messages` (chat) or `text` |
| DPO | `trl dpo` | `prompt`, `chosen`, `rejected` |
| GRPO | `trl grpo` | `prompt` + reward functions |
| KTO | `trl kto` | `prompt`, `completion`, `label` |

### Minimal LoRA SFT config

```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
datasets:
  - path: /path/to/data
output_dir: ./output
num_train_epochs: 3
learning_rate: 0.0002
bf16: true
use_peft: true
lora_r: 16
lora_alpha: 32
report_to: none
```

### Key field names

- `model_name_or_path` — HuggingFace model ID
- `num_train_epochs` — number of epochs
- `per_device_train_batch_size` — batch size per GPU
- `max_length` — max sequence length
- `use_peft` — enable LoRA
- `lora_r`, `lora_alpha` — LoRA configuration
- `accelerate_config` — distributed training (zero2, fsdp2, etc.)

### Container image

`docker.io/huggingface/trl:1.5.0` — official HuggingFace image with all dependencies.

## asynth — Synthesis Engine

asynth is the synthesis engine for generating training data. Install: `pip install asynth`.

### Core API

```python
from asynth import synthesize, SynthesisConfig, LiteLLMInferenceConfig
from asynth.configs.params.synthesis_params import GeneralSynthesisParams

config = SynthesisConfig(
    num_samples=100,
    output_path="./output.jsonl",
    inference_config=LiteLLMInferenceConfig(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        max_concurrency=16,
    ),
    strategy_params=GeneralSynthesisParams(),
)
results = synthesize(config)  # returns list[dict]
```

### LiteLLMInferenceConfig

```python
LiteLLMInferenceConfig(
    model="openai/gpt-4o-mini",    # LiteLLM format — 100+ providers
    temperature=1.0,
    max_tokens=None,
    top_p=None,
    max_concurrency=16,
    num_retries=3,
    api_base=None,                  # custom API endpoint
    api_key=None,
)
```

### Teacher model providers (via LiteLLM)

Supports 100+ providers: OpenAI, Anthropic, Google, vLLM (`hosted_vllm/`), Ollama (`ollama/`), Azure, Bedrock, Cohere, Mistral, Together, Groq, OpenRouter, and more.

### GeneralSynthesisParams — Attribute types

| Attribute Type | Class | Purpose |
|---|---|---|
| Sampled | `SampledAttribute` | Categorical variable sampling with rates |
| Generated | `GeneratedAttribute` | Single-turn LLM-generated outputs |
| Multi-turn | `MultiTurnAttribute` | Multi-round conversation synthesis |
| Transformed | `TransformedAttribute` | Post-hoc transforms (string/list/dict/chat) |

### Data sources

| Source | Class | Formats |
|---|---|---|
| Datasets | `DatasetSource` | JSONL, CSV, Parquet, HuggingFace |
| Documents | `DocumentSource` | PDF, DOCX, TXT (token-based segmentation) |
| Examples | `ExampleSource` | Inline example dicts |

### Pipeline execution order

1. **Dataset planning** — sample attributes, load sources, create rows
2. **Generated attribute synthesis** — batch LLM calls for single-turn outputs
3. **Conversation synthesis** — turn-by-turn multi-turn generation with tool-call loops
4. **Attribute transformation** — apply string/list/dict/chat transforms
5. **Quality checking** — structural validation on conversation outputs
6. **Save** — write JSONL if `output_path` is set

### Output format

`synthesize()` returns `list[dict]`, one dict per sample. Keys come from attribute IDs. When `output_path` is set, also saves as JSONL.

### Judges API

```python
from asynth import judge, create_judge, JudgeConfig

# Quick judge with built-in template
results = judge("generic/safety", data=[{"response": "..."}], model="openai/gpt-4o-mini")

# Custom judge
config = JudgeConfig.from_path("my_judge.yaml")
j = create_judge(config, inference_config=LiteLLMInferenceConfig(model="openai/gpt-4o"))
results = j.judge(data)
```

Built-in judge categories: generic (safety, truthfulness, instruction_following), code (quality, correctness, security), doc_qa (completeness, groundedness, relevance).

### Credentials

API keys are resolved in order:
1. `api_key` in job config (per-job override)
2. `AMORTIZED_LLM_API_KEY` environment variable
3. Provider-specific env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) — read by LiteLLM automatically

## The Amortization Workflow

The core workflow to replace expensive frontier model calls with smaller, customized models:

1. **Analyze** — Understand the user's agent task, identify which LLM calls are expensive and repetitive
2. **Generate data (SDG)** — Use a teacher model (e.g. GPT-4o) to generate training data via asynth
3. **Fine-tune (LoRA SFT)** — Train a small model (e.g. Qwen 1.5B) on the generated data using TRL
4. **Evaluate** — Compare the fine-tuned model against the original frontier model on the task
5. **Deploy** — Replace the frontier model call with the smaller, cheaper fine-tuned model

This process "amortizes" the cost of the frontier model across many future inferences.
