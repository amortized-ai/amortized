# Amortized

**Perfect your AI agent workflows — swap frontier models for smaller, cheaper, custom models without sacrificing quality.**

---

Amortized is a studio for optimizing AI agent pipelines. It lets you take workflows built on large or frontier API models and systematically replace them with smaller, more efficient, customized models — all while maintaining agent quality and task accuracy.

Built on the [Red Hat AI Innovation Training Hub](https://github.com/redhat-ai-services/training-hub) and [asynth](https://github.com/amortized-ai/asynth) for model customization, Amortized provides a fully open-source, on-premises experience that requires no prior model customization expertise.

## What It Does

- **Agent Workflow Analysis** — Import your existing agent workflows and identify which model calls are candidates for optimization
- **Synthetic Data Generation** — Automatically generate task-specific training data from frontier model outputs using asynth
- **Model Customization** — Fine-tune smaller models on your agent's specific tasks using the Red Hat AI Innovation Training Hub
- **Quality Evaluation** — Validate that customized models maintain accuracy and agent behavior through automated benchmarking
- **Iterative Refinement** — Continuously improve custom models with feedback loops until they match frontier model performance

## Key Features

- **UI Dashboard** — Visual interface for managing workflows, monitoring training jobs, and comparing model performance
- **Agent Chat Interface** — Test and interact with your agent workflows side-by-side across model configurations
- **Job & Artifact Tracking** — Track training runs, model versions, evaluation results, and deployment artifacts
- **No Expertise Required** — Guided experience that abstracts away the complexity of model customization
- **Fully On-Prem** — Everything runs on your infrastructure — no data leaves your environment
- **Open Source** — Apache 2.0 licensed, extensible, and community-driven

## Why "Amortized"

In the same way amortization spreads a large cost over time, Amortized spreads the capability of expensive frontier models across cheaper, specialized ones — reducing your per-inference cost while preserving the quality your agents depend on.

## Getting Started

### 1. Install

```bash
git clone https://github.com/amortized-ai/amortized.git
cd amortized/server
pip install -e .
```

### 2. Start the Server

```bash
amortized up
```

Verify it's running: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### 3. First-Time Setup (optional)

```bash
amortized init
```

Walks you through configuring API keys and GPU backends.

### 4. Generate Synthetic Data

```bash
amortized submit sdg --confirm \
  --recipe base/sdg \
  --set config.model=openai/gpt-4o-mini
```

### 5. Train a Model

```bash
amortized submit training --confirm \
  --set config.algorithm=lora_sft \
  --set config.model_path=Qwen/Qwen2.5-1.5B-Instruct \
  --set config.data_path=./data.jsonl
```

### 6. Connect via MCP

Add this to your Claude Code or Cursor MCP config:

```json
{
  "mcpServers": {
    "amortized": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## License

[Apache License 2.0](LICENSE)
