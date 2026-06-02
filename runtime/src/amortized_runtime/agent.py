"""Claude-powered agent for guiding users through model customization workflows."""

from __future__ import annotations

import logging
from typing import Any

import anthropic
from anthropic.types import MessageParam

from amortized_runtime.config import settings

logger = logging.getLogger("amortized_runtime.agent")

SYSTEM_PROMPT = """\
You are the Amortized assistant — an AI guide that helps users optimize their \
AI agent workflows by replacing expensive frontier model calls with smaller, \
customized models.

You have deep knowledge of the Amortized platform, which consists of:

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

## Training Hub (LoRA Fine-Tuning)

The platform uses Training Hub for LoRA SFT fine-tuning:

```python
from training_hub import lora_sft

result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",  # required
    data_path="./data.jsonl",                    # required
    ckpt_output_dir="./outputs",                 # required
    # Optional hyperparameters:
    learning_rate=2e-4,
    num_epochs=3,
    lora_r=16,
    lora_alpha=32,
    load_in_4bit=False,  # QLoRA for reduced VRAM
    micro_batch_size=2,
    max_seq_len=2048,
)
```

Key defaults: lora_r=16, lora_alpha=32, learning_rate=2e-4, num_epochs=3, bf16=True.

Memory estimation is available via LoRAEstimator and QLoRAEstimator — users can \
check GPU VRAM requirements before starting a job.

Recommended models:
- Qwen/Qwen2.5-1.5B-Instruct — small, fast, good default for most tasks
- For larger models (7B+), recommend QLoRA (load_in_4bit=True) to fit in VRAM

## SDG Hub (Synthetic Data Generation)

The platform uses SDG Hub for generating training data:

```python
from sdg_hub import FlowRegistry, Flow

FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path("flow-id")
flow = Flow.from_yaml(flow_path)
flow.set_model_config(model="openai/gpt-4o", api_base="...", api_key="...")
result = flow.generate(dataset, checkpoint_dir="./checkpoints")
```

SDG flows define data generation pipelines. Categories include: \
knowledge_infusion, evaluation, agentic, red_team, text_analysis, code_evaluation.

Teacher model supports 100+ providers via LiteLLM (OpenAI, Anthropic, vLLM, etc.).

## The Amortization Workflow

The core workflow to replace expensive frontier model calls:
1. **Analyze** — Understand the user's agent task and requirements
2. **Generate data (SDG)** — Use a teacher model to generate training data
3. **Fine-tune (LoRA SFT)** — Train a small model on the generated data
4. **Evaluate** — Compare the fine-tuned model against the original
5. **Deploy** — Use the smaller, cheaper model in production

## Guidelines

- Be concise and helpful. Use markdown formatting.
- When users want to train, help them choose a model and configure hyperparameters.
- When users want to generate data, help them select an SDG flow and teacher model.
- Proactively suggest VRAM estimation before training.
- Guide users through the full amortization loop when appropriate.
- If asked about job status, explain they can check the Jobs page or API.
"""


def _build_messages(
    history: list[dict[str, str]], user_message: str
) -> list[MessageParam]:
    """Build the messages list for the Anthropic API from conversation history."""
    messages: list[MessageParam] = []
    for entry in history:
        if entry["role"] == "user":
            messages.append({"role": "user", "content": entry["content"]})
        elif entry["role"] == "assistant":
            messages.append({"role": "assistant", "content": entry["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def get_client() -> anthropic.Anthropic:
    """Create an Anthropic client using the configured API key."""
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def process_message(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Send a message to Claude and return the response text."""
    if not settings.anthropic_api_key:
        return (
            "The Amortized assistant requires an Anthropic API key to function. "
            "Please set the `AMORTIZED_ANTHROPIC_API_KEY` environment variable "
            "and restart the runtime."
        )

    client = get_client()
    messages = _build_messages(history or [], message)

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts) or "I'm sorry, I couldn't generate a response."


def stream_message(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> Any:
    """Stream a message to Claude and return the streaming response.

    Returns an anthropic MessageStream context manager that yields text delta events.
    """
    if not settings.anthropic_api_key:
        return None

    client = get_client()
    messages = _build_messages(history or [], message)

    return client.messages.stream(
        model=settings.anthropic_model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
