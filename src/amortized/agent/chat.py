"""OpenAI function-calling agent for guiding users through model customization."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from amortized.agent.protocol import EventType, StreamEvent
from amortized.agent.schemas import TOOLS
from amortized.agent.tools import execute_tool, tool_result_summary
from amortized.config import settings

if TYPE_CHECKING:
    from amortized.db.repository import Repository

logger = logging.getLogger("amortized.agent")

SYSTEM_PROMPT = """\
You are the Amortized Studio assistant — a friendly AI expert embedded in a \
web dashboard that helps users replace expensive frontier model calls with \
smaller, fine-tuned models.

## YOUR ROLE

You handle ALL technical work on behalf of the user. The user interacts with \
you through a chat interface in their browser — they do NOT have terminal \
access and cannot run commands. You have tools to call the Amortized runtime \
API on their behalf.

## CRITICAL RULES
- NEVER ask the user to run commands, scripts, or code.
- NEVER reference project files, directories, or scripts.
- Always use your tools to interact with the runtime API.
- Present results in user-friendly language with markdown formatting.
- When proposing a job (training or SDG), ALWAYS use the propose_action tool \
so the user can review the configuration and click a button to confirm. \
Do NOT call submit_training_job or submit_sdg_job directly unless the user \
has explicitly confirmed they want to proceed.
- Before submitting an SDG job, ALWAYS check list_api_keys first. If the \
provider key is missing, ask for it and store via add_api_key before proceeding.
- When you need to create seed data for an SDG flow, use the create_dataset \
tool. DO NOT reference files that don't exist.
- Always preview the dataset after creating it so the user can verify the content.
- For knowledge tuning flows, create documents that contain the factual \
information the user wants to teach the model. Each document should be \
substantial (100+ words).
- You CANNOT load models, run inference, or test trained models. There is no \
inference endpoint.
- You CAN preview artifacts and read metrics. Use read_artifact_preview to \
inspect generated data, training metrics, and other text-based outputs.
- After SDG completes, use read_artifact_preview to check data quality. Show \
the user a few example rows and assess whether the data looks good for training.
- After SDG completes, the output data usually needs to be converted to messages \
format before training. Use convert_dataset to transform SDG output \
(question/answer pairs) into the messages format that TRL expects. \
Always convert BEFORE proposing a training job.
- After training completes, check the training metrics to assess convergence \
(look at final loss, whether loss plateaued, etc.) and give recommendations. \
Tell the user they can download the adapter from the Artifacts tab to test it.
- Take things ONE STEP AT A TIME. Never dump a full plan or ask more than \
1-2 questions at once.
- Be conversational, not a project manager. Short, focused responses.
- First understand what the user wants (1-2 clarifying questions max).
- Then propose the FIRST step only (e.g., an SDG job). Don't mention future \
steps yet.
- After each job completes, look at the results BEFORE deciding the next step.
- Check job status using your tools when the user asks — don't just tell them \
to check the Jobs page.
- When a job finishes, examine the outputs (metrics, artifacts) and make \
data-driven recommendations for the next step.
- Keep responses SHORT. 3-5 sentences max for most messages. Only get detailed \
when explaining results.
- Use sensible defaults — don't ask the user about lora_r, learning_rate, \
batch_size, etc. unless they bring it up. Just pick good values.
- You're a friendly expert guide, not a requirements-gathering bot.
- When you want the user to choose between 2-4 options (e.g. model size, \
approach, SDG flow), use the present_options tool to render clickable cards \
instead of listing them in text.

## AVAILABLE TOOLS
- **list_sdg_flows**: Discover available SDG flows for data generation
- **submit_sdg_job**: Submit an SDG job (prefer propose_action for user confirmation)
- **submit_training_job**: Submit a LoRA SFT training job (prefer propose_action)
- **check_job_status**: Check status of a running or completed job
- **get_job_metrics**: Get training metrics (loss, LR, epoch per step)
- **list_jobs**: List all jobs with optional filters
- **estimate_vram**: Estimate GPU VRAM for a training configuration
- **read_artifact_preview**: Preview artifact contents (data samples, metrics, logs)
- **create_dataset**: Create a JSONL seed dataset file for SDG flows
- **preview_dataset**: Preview the first few rows of a dataset file
- **convert_dataset**: Convert SDG output to messages format for training
- **judge_data**: Judge data quality using asynth's built-in judges (e.g. safety, correctness)
- **list_judge_templates**: List available judge templates
- **list_api_keys**: Check which LLM provider keys are configured
- **add_api_key**: Store a provider API key (encrypted, persists)
- **propose_action**: Propose a job for user confirmation (renders as a button)
- **present_options**: Present 2-4 options for the user to choose from (renders as cards)

## TRL KNOWLEDGE (LoRA SFT)

Training runs via the HuggingFace TRL CLI (`trl sft --config config.yaml`). \
Key parameters:
- **model_name_or_path** (required): HuggingFace model ID (e.g. "Qwen/Qwen2.5-1.5B-Instruct")
- **datasets** (required): Path to training data in JSONL format
- **output_dir** (required): Output directory for checkpoints
- Sensible defaults: learning_rate=2e-4, num_train_epochs=3, \
per_device_train_batch_size=2, max_length=2048, lora_r=16, lora_alpha=32, \
use_peft=true, bf16=true.

Recommended models:
- **Qwen/Qwen2.5-1.5B-Instruct** — small, fast, good default
- For 7B+ models, recommend QLoRA (load_in_4bit=true)

Output: HuggingFace PEFT format (adapter_model.safetensors + adapter_config.json). \
Metrics written as training_metrics.jsonl with per-step loss, LR, epoch.

## ASYNTH KNOWLEDGE (Synthetic Data Generation)

asynth generates synthetic training data using a teacher model. It uses an attribute-based \
system: you define what kind of data to generate via strategy_params, and asynth orchestrates \
the LLM calls to produce it.

Key concepts:
- **strategy_params**: Defines attributes for generation via GeneralSynthesisParams
- **model**: Teacher model in LiteLLM format (e.g. openai/gpt-4o, hosted_vllm/model-name)
- **num_samples**: How many samples to generate

strategy_params dict keys (IMPORTANT — use these exact field names):
- **sampled_attributes**: list of categorical variables with sample rates
- **generated_attributes**: list of LLM-generated outputs with instruction_messages
- **multiturn_attributes**: multi-round conversation config
- **transformed_attributes**: post-hoc transforms (string/list/dict/chat)
- **passthrough_attributes**: list of attribute IDs to include in output
- **input_examples**: few-shot examples for generation
- **input_data**: external dataset sources
- **input_documents**: document sources (PDF, DOCX, TXT)

Do NOT use "attributes" as a key — always use the full names above.

Data sources (use input_data or input_documents in SDG config):
- **DatasetSource**: Feed existing datasets (JSONL, CSV, Parquet, HuggingFace) via input_data
- **DocumentSource**: Feed documents (PDF, DOCX, TXT) via input_documents for \
document-grounded generation

Environment awareness:
- For tool-use conversation generation, MultiTurnAttribute supports environment configs \
that define available tools and their schemas for realistic tool-call conversations.

Judges (quality assessment):
- After SDG completes, use judge_data to assess quality before training
- Built-in templates: generic/safety, generic/truthfulness, generic/instruction_following, \
code/quality, code/correctness, code/security, doc_qa/completeness, doc_qa/groundedness, \
doc_qa/relevance
- Use list_judge_templates to discover all available templates
- Judging uses a separate LLM call — specify a model (e.g. openai/gpt-4o-mini)

Workflow for SDG:
1. Understand what kind of training data the user needs
2. Propose an SDG job with the teacher model and num_samples
3. For advanced use cases, configure strategy_params with attribute definitions
4. Use input_data to feed existing datasets, input_documents for PDFs/docs
5. The job runs asynth's synthesis pipeline and outputs a JSONL file
6. After completion, use judge_data to assess quality before proceeding to training

Teacher model support: 100+ providers via LiteLLM — OpenAI, Anthropic, Google, \
vLLM (hosted_vllm/), Ollama (ollama/), Azure, and more.

## API KEY MANAGEMENT
Before proposing ANY SDG or eval job:
1. Call list_api_keys to check which provider keys are configured
2. If the target provider's key is missing, ask the user for it
3. Store it via add_api_key — it persists across jobs and restarts
4. NEVER tell users to set environment variables or edit files
5. NEVER put api_key directly in job configs — use the managed store

The key is encrypted at rest, redacted in all API responses, and injected \
automatically into job containers at runtime.

## TIPS
- Use estimate_vram before proposing a training job so you can advise on config
- For training, recommend starting with Qwen 1.5B unless the task requires more
- When the user asks about job status, USE your tools — don't tell them to check
"""


def _build_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from settings."""
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _history_to_messages(
    history: list[dict[str, str]] | None,
) -> list[ChatCompletionMessageParam]:
    """Convert stored conversation history to OpenAI message format."""
    msgs: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if history:
        for entry in history:
            role = entry["role"]
            if role == "user":
                msgs.append({"role": "user", "content": entry["content"]})
            elif role == "assistant":
                msgs.append({"role": "assistant", "content": entry["content"]})
    return msgs


# ---------------------------------------------------------------------------
# Non-streaming (synchronous) path
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Result of a complete agent turn."""

    text: str
    proposed_action: dict[str, Any] | None = None
    presented_options: dict[str, Any] | None = None


async def process_message(
    message: str,
    history: list[dict[str, str]] | None = None,
    *,
    repo: Repository | None = None,
) -> AgentResult:
    """Run the full agentic loop (non-streaming) and return the final response."""
    client = _build_client()
    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})

    proposed_action: dict[str, Any] | None = None
    presented_options: dict[str, Any] | None = None
    max_iterations = 20

    for _ in range(max_iterations):
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            tools=TOOLS,  # type: ignore[arg-type]
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            messages.append(choice.message)  # type: ignore[arg-type]

            for tc in choice.message.tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                args = json.loads(fn.arguments) if fn.arguments else {}
                result = await execute_tool(fn.name, args, repo)

                if result.get("__proposed_action__"):
                    proposed_action = {
                        "type": result["action_type"],
                        "config": result["config"],
                        "label": result["label"],
                    }
                    tool_content = json.dumps(
                        {"status": "proposed", "message": "Action proposed to user"}
                    )
                elif result.get("__present_options__"):
                    presented_options = {
                        "prompt": result["prompt"],
                        "options": result["options"],
                    }
                    tool_content = json.dumps(
                        {"status": "options_presented", "message": "Options shown to user"}
                    )
                else:
                    tool_content = json.dumps(result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    }
                )
            continue

        return AgentResult(
            text=choice.message.content or "",
            proposed_action=proposed_action,
            presented_options=presented_options,
        )

    return AgentResult(text="I've reached my processing limit. Please try again.")


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------


async def stream_message(
    message: str,
    history: list[dict[str, str]] | None = None,
    *,
    repo: Repository | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run the agentic loop with streaming, yielding events as they happen."""
    client = _build_client()
    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})

    proposed_action: dict[str, Any] | None = None
    presented_options: dict[str, Any] | None = None
    full_text = ""
    max_iterations = 20

    try:
        for _ in range(max_iterations):
            stream = await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=TOOLS,  # type: ignore[arg-type]
                stream=True,
            )
            assert hasattr(stream, "__aiter__"), "Expected async stream"

            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                if delta.content:
                    content_parts.append(delta.content)
                    full_text += delta.content
                    yield StreamEvent(
                        type=EventType.delta, data={"text": delta.content}
                    )

                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_chunk.id:
                            tool_calls_acc[idx]["id"] = tc_chunk.id
                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                tool_calls_acc[idx]["name"] = tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc_chunk.function.arguments

            if tool_calls_acc and finish_reason == "tool_calls":
                tool_calls_list = []
                for idx in sorted(tool_calls_acc.keys()):
                    tc_data = tool_calls_acc[idx]
                    tool_calls_list.append(
                        {
                            "id": tc_data["id"],
                            "type": "function",
                            "function": {
                                "name": tc_data["name"],
                                "arguments": tc_data["arguments"],
                            },
                        }
                    )

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls_list,
                }
                content_str = "".join(content_parts)
                if content_str:
                    assistant_msg["content"] = content_str
                messages.append(assistant_msg)  # type: ignore[arg-type]

                for tc_data in tool_calls_list:
                    name = tc_data["function"]["name"]
                    yield StreamEvent(
                        type=EventType.thinking, data={"tool": name}
                    )

                    raw_args = tc_data["function"]["arguments"]
                    args = json.loads(raw_args) if raw_args else {}
                    result = await execute_tool(name, args, repo)

                    if result.get("__proposed_action__"):
                        proposed_action = {
                            "type": result["action_type"],
                            "config": result["config"],
                            "label": result["label"],
                        }
                        tool_content = json.dumps(
                            {"status": "proposed", "message": "Action proposed to user"}
                        )
                        summary = "Action proposed for confirmation"
                    elif result.get("__present_options__"):
                        presented_options = {
                            "prompt": result["prompt"],
                            "options": result["options"],
                        }
                        tool_content = json.dumps(
                            {"status": "options_presented", "message": "Options shown to user"}
                        )
                        summary = "Options presented to user"
                    else:
                        tool_content = json.dumps(result)
                        summary = tool_result_summary(name, result)

                    yield StreamEvent(
                        type=EventType.tool_result,
                        data={"tool": name, "summary": summary},
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_data["id"],
                            "content": tool_content,
                        }
                    )

                content_parts = []
                continue

            break

        if proposed_action:
            yield StreamEvent(type=EventType.action, data=proposed_action)

        if presented_options:
            yield StreamEvent(type=EventType.options, data=presented_options)

        yield StreamEvent(type=EventType.done, data={"full_text": full_text})

    except Exception as exc:
        logger.exception("Streaming agent error")
        yield StreamEvent(
            type=EventType.error,
            data={"error": f"Agent error: {exc}"},
        )
