# Amortized — AI Model Customization Studio

## Project Purpose

Amortized is a fully open-source, on-premises studio for optimizing AI agent workflows. It replaces expensive frontier model calls with smaller, customized models — without sacrificing quality. Built on [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) (LoRA fine-tuning) and [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub) (synthetic data generation).

## Monorepo Layout

```
amortized/
├── CLAUDE.md              # This file — project conventions
├── factory.md             # Factory configuration and eval dimensions
├── runtime/               # Python FastAPI backend
│   ├── pyproject.toml
│   ├── src/amortized_runtime/
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI app entry point
│   │   └── config.py      # Settings via pydantic-settings
│   └── tests/
├── studio/                # Next.js React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   └── src/app/
├── docker/                # Container definitions
│   └── Dockerfile.runtime
├── eval/                  # Factory eval harness
│   └── score.py
└── .github/workflows/     # CI pipelines
    └── ci.yml
```

## Dev Commands

### Runtime (Python FastAPI backend)

```bash
cd runtime
pip install -e '.[dev]'
uvicorn amortized_runtime.main:app --reload
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

## Python Conventions (runtime/)

- **Linter**: `ruff check src/ tests/` — enforced in CI
- **Formatter**: `ruff format src/ tests/`
- **Type checker**: `mypy src/` — strict mode enabled
- **Tests**: `pytest` — use `pytest-asyncio` for async tests
- **Package structure**: src layout (`src/amortized_runtime/`)
- **Settings**: Use `pydantic-settings` for configuration (env vars, .env files)
- **API versioning**: All endpoints under `/api/v1/`

## TypeScript Conventions (studio/)

- **Linter**: `npx eslint .`
- **Type checking**: TypeScript strict mode enabled in tsconfig.json
- **Styling**: Tailwind CSS with dark theme
- **Components**: shadcn/ui component library
- **App Router**: Next.js 15 App Router (src/app/ directory)

## Training Hub Integration Patterns

Training Hub provides LoRA fine-tuning via a Python API:

```python
from training_hub import lora_sft

result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    # Optional: learning_rate, num_epochs, lora_r, lora_alpha, etc.
)
```

- Only 3 required params: `model_path`, `data_path`, `ckpt_output_dir`
- Metrics written to `training_metrics.jsonl` in ckpt_output_dir (per-step loss, LR, epoch)
- Memory estimation via `LoRAEstimator` / `QLoRAEstimator`
- No YAML configs needed — accepts Python kwargs directly

## SDG Hub Integration Patterns

SDG Hub provides synthetic data generation via a Python API:

```python
from sdg_hub import FlowRegistry, Flow

FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path("flow-id")
flow = Flow.from_yaml(flow_path)
flow.set_model_config(model="openai/gpt-4o", api_base="...", api_key="...")
result = flow.generate(dataset, checkpoint_dir="./checkpoints")
```

- `FlowRegistry.discover_flows()` lists available SDG flows
- `Flow.from_yaml()` loads a flow definition
- `flow.set_model_config()` configures the teacher model (supports 100+ providers via LiteLLM)
- `flow.generate()` runs the pipeline, returns enriched dataset
- Progress via `FlowCheckpointer.get_progress_info()`
