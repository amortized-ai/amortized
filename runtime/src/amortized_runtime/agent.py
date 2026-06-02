"""OpenAI function-calling agent for guiding users through model customization."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from amortized_runtime.config import settings
from amortized_runtime.tools import TOOLS, execute_tool, tool_result_summary

logger = logging.getLogger("amortized_runtime.agent")

SYSTEM_PROMPT = """\
You are the Amortized Studio assistant — an AI concierge embedded in a web \
dashboard that helps users optimize their AI agent workflows by replacing \
expensive frontier model calls with smaller, fine-tuned models.

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

## THE AMORTIZATION WORKFLOW
1. **Understand** the user's agent task and identify expensive LLM calls
2. **Generate training data** using SDG (synthetic data generation) with a \
teacher model
3. **Fine-tune** a small model (e.g. Qwen 1.5B) on the generated data
4. **Evaluate** the fine-tuned model against the original
5. **Deploy** the smaller, cheaper model

## AVAILABLE TOOLS
- **list_sdg_flows**: Discover available SDG flows for data generation
- **submit_sdg_job**: Submit an SDG job (prefer propose_action for user confirmation)
- **submit_training_job**: Submit a LoRA SFT training job (prefer propose_action)
- **check_job_status**: Check status of a running or completed job
- **get_job_metrics**: Get training metrics (loss, LR, epoch per step)
- **list_jobs**: List all jobs with optional filters
- **estimate_vram**: Estimate GPU VRAM for a training configuration
- **propose_action**: Propose a job for user confirmation (renders as a button)

## TRAINING HUB KNOWLEDGE (LoRA SFT)

Training Hub provides LoRA fine-tuning. Key parameters:
- **model_path** (required): HuggingFace model ID (e.g. "Qwen/Qwen2.5-1.5B-Instruct")
- **data_path** (required): Path to training data in JSONL format
- **ckpt_output_dir** (required): Output directory for checkpoints
- **learning_rate**: Default 2e-4
- **num_epochs**: Default 3
- **micro_batch_size**: Default 2
- **max_seq_len**: Default 2048
- **lora_r**: LoRA rank — default 16, higher = more expressive
- **lora_alpha**: Scaling factor — default 32, typically 2x lora_r
- **load_in_4bit**: Enable QLoRA for reduced VRAM — fits 7B+ on 24GB GPU

Recommended models:
- **Qwen/Qwen2.5-1.5B-Instruct** — small, fast, good default
- For 7B+ models, recommend QLoRA (load_in_4bit=true)
- A single 24GB GPU can fine-tune 7B with LoRA, 20B+ with QLoRA

Output: HuggingFace PEFT format (adapter_model.safetensors + adapter_config.json). \
Metrics written as training_metrics.jsonl with per-step loss, LR, epoch.

## SDG HUB KNOWLEDGE (Synthetic Data Generation)

SDG Hub generates training data using teacher models. Flow categories:
- **knowledge_infusion**: Q&A generation, summaries, knowledge extraction
- **evaluation**: RAG evaluation, answer quality assessment
- **agentic**: MCP distillation, agent behavior datasets
- **red_team**: Adversarial prompt generation
- **text_analysis**: Classification, sentiment, text transformation
- **code_evaluation**: Code quality, bug detection datasets

Teacher model support: 100+ providers via LiteLLM — OpenAI, Anthropic, Google, \
vLLM (hosted_vllm/), Ollama (ollama/), Azure, and more.

## TIPS
- Use estimate_vram before proposing a training job so you can advise on \
configuration
- Always explain what you're doing and why in plain language
- When listing flows or jobs, summarize the results concisely
- For training, recommend starting with Qwen 1.5B unless the task requires more
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


async def process_message(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> AgentResult:
    """Run the full agentic loop (non-streaming) and return the final response."""
    client = _build_client()
    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})

    proposed_action: dict[str, Any] | None = None
    max_iterations = 20

    for _ in range(max_iterations):
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            tools=TOOLS,  # type: ignore[arg-type]
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # Append the assistant message with tool calls
            messages.append(choice.message)  # type: ignore[arg-type]

            for tc in choice.message.tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                args = json.loads(fn.arguments) if fn.arguments else {}
                result = await execute_tool(fn.name, args)

                if result.get("__proposed_action__"):
                    proposed_action = {
                        "type": result["action_type"],
                        "config": result["config"],
                        "label": result["label"],
                    }
                    tool_content = json.dumps(
                        {"status": "proposed", "message": "Action proposed to user"}
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

        # Model responded with text — we're done
        return AgentResult(
            text=choice.message.content or "",
            proposed_action=proposed_action,
        )

    return AgentResult(text="I've reached my processing limit. Please try again.")


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """An event emitted during streaming."""

    type: str  # "thinking", "tool_result", "delta", "action", "done", "error"
    data: dict[str, Any] = field(default_factory=dict)


async def stream_message(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run the agentic loop with streaming, yielding events as they happen."""
    client = _build_client()
    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})

    proposed_action: dict[str, Any] | None = None
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

            # Accumulate the streamed response
            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Stream text deltas
                if delta.content:
                    content_parts.append(delta.content)
                    full_text += delta.content
                    yield StreamEvent(type="delta", data={"text": delta.content})

                # Accumulate tool call chunks
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
                                tool_calls_acc[idx]["arguments"] += (
                                    tc_chunk.function.arguments
                                )

            # If we got tool calls, execute them and continue
            if tool_calls_acc and finish_reason == "tool_calls":
                # Build the assistant message with tool calls for history
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

                # Execute each tool call
                for tc_data in tool_calls_list:
                    name = tc_data["function"]["name"]
                    yield StreamEvent(type="thinking", data={"tool": name})

                    raw_args = tc_data["function"]["arguments"]
                    args = json.loads(raw_args) if raw_args else {}
                    result = await execute_tool(name, args)

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
                    else:
                        tool_content = json.dumps(result)
                        summary = tool_result_summary(name, result)

                    yield StreamEvent(
                        type="tool_result",
                        data={"tool": name, "summary": summary},
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_data["id"],
                            "content": tool_content,
                        }
                    )

                # Reset content for next iteration
                content_parts = []
                continue

            # Model is done — no more tool calls
            break

        if proposed_action:
            yield StreamEvent(type="action", data=proposed_action)

        yield StreamEvent(type="done", data={"full_text": full_text})

    except Exception as exc:
        logger.exception("Streaming agent error")
        yield StreamEvent(
            type="error",
            data={"error": f"Agent error: {exc}"},
        )
