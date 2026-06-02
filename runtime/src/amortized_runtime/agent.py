"""Rule-based agent for guiding users through model customization workflows."""

import re
from collections.abc import Callable

from amortized_runtime.models import AgentResponse, SuggestedAction


class AmortizedAgent:
    """Rule-based agent that matches user intents and returns structured responses."""

    def __init__(self) -> None:
        _c = re.compile
        _i = re.IGNORECASE
        # Order matters: SDG before training so "generate data" matches SDG
        self._intents: list[
            tuple[re.Pattern[str], Callable[[str], AgentResponse]]
        ] = [
            (_c(r"(?:generat|synthe|sdg|data\s*gen)", _i), self._handle_sdg),
            (_c(r"(?:fine[- ]?tun|train|lora|sft|custom)", _i), self._handle_training),
            (_c(r"(?:status|progress|running|check\s*(?:on|my))", _i), self._handle_status),
            (_c(r"(?:vram|memory|gpu|ram|estimate|how\s*much)", _i), self._handle_estimation),
            (_c(r"(?:flow|pipeline|available|capabilit|feature)", _i), self._handle_flows),
            (_c(r"(?:artifact|output|result|download|adapter)", _i), self._handle_artifacts),
            (_c(r"(?:help|hello|hi|hey|who|how\s*do)", _i), self._handle_help),
        ]

    def _handle_training(self, message: str) -> AgentResponse:
        model = "Qwen/Qwen2.5-1.5B-Instruct"
        if "llama" in message.lower():
            model = "meta-llama/Llama-3.1-8B-Instruct"
        elif "mistral" in message.lower():
            model = "mistralai/Mistral-7B-Instruct-v0.3"

        return AgentResponse(
            message=(
                f"I recommend using **LoRA SFT** with `{model}`. "
                "Here's a suggested configuration:\n\n"
                f"- **Model**: `{model}`\n"
                "- **LoRA rank**: 16\n"
                "- **Learning rate**: 2e-4\n"
                "- **Epochs**: 3\n\n"
                "You can adjust these parameters based on your dataset size "
                "and quality requirements. Would you like me to estimate the "
                "VRAM requirements first?"
            ),
            suggested_action=SuggestedAction(
                type="create_training_job",
                label="Create Training Job",
                config={
                    "model_path": model,
                    "data_path": "",
                    "ckpt_output_dir": "./outputs",
                    "learning_rate": 2e-4,
                    "num_epochs": 3,
                    "lora_r": 16,
                    "lora_alpha": 32,
                },
            ),
            context={
                "model_info": {
                    "name": model,
                    "type": "causal_lm",
                    "recommended_for": "general fine-tuning",
                }
            },
        )

    def _handle_sdg(self, message: str) -> AgentResponse:
        return AgentResponse(
            message=(
                "I can help you generate synthetic training data using SDG Hub. "
                "Here's what I recommend:\n\n"
                "1. **Choose a flow** — SDG flows define the data generation pipeline\n"
                "2. **Provide seed data** — A small dataset of examples to build from\n"
                "3. **Configure the teacher model** — The LLM that generates new samples\n\n"
                "Would you like to see the available SDG flows?"
            ),
            suggested_action=SuggestedAction(
                type="create_sdg_job",
                label="Create SDG Job",
                config={
                    "flow_id": "",
                    "dataset_path": "",
                    "model": "openai/gpt-4o",
                },
            ),
            context={
                "workflow": "sdg",
                "next_step": "Select an SDG flow from the Flows page",
            },
        )

    def _handle_status(self, message: str) -> AgentResponse:
        return AgentResponse(
            message=(
                "You can check the status of your jobs on the **Jobs** page. "
                "Each job shows its current state (pending, running, completed, "
                "or failed) along with real-time metrics.\n\n"
                "For training jobs, you'll see loss and learning rate curves. "
                "For SDG jobs, you'll see generation progress."
            ),
            suggested_action=SuggestedAction(
                type="view_jobs",
                label="View Jobs",
                config={},
            ),
        )

    def _handle_estimation(self, message: str) -> AgentResponse:
        return AgentResponse(
            message=(
                "I can estimate the GPU VRAM requirements for your training run. "
                "The estimate depends on:\n\n"
                "- **Model size** — larger models need more VRAM\n"
                "- **LoRA rank** — higher rank = more trainable parameters\n"
                "- **Batch size** — larger batches need more memory\n"
                "- **Sequence length** — longer sequences use more memory\n"
                "- **QLoRA** — 4-bit quantization reduces VRAM by ~60%\n\n"
                "Go to **New Job → Estimate VRAM** to run a pre-flight check."
            ),
            suggested_action=SuggestedAction(
                type="estimate_memory",
                label="Estimate VRAM",
                config={
                    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "lora_r": 16,
                    "batch_size": 2,
                    "max_seq_len": 2048,
                    "load_in_4bit": False,
                },
            ),
        )

    def _handle_flows(self, message: str) -> AgentResponse:
        return AgentResponse(
            message=(
                "Amortized supports two main workflows:\n\n"
                "### Synthetic Data Generation (SDG)\n"
                "Generate training data from seed examples using a teacher model. "
                "Multiple flow templates are available for different use cases.\n\n"
                "### LoRA Fine-Tuning\n"
                "Fine-tune models with LoRA adapters for efficient customization. "
                "Supports QLoRA for reduced VRAM usage.\n\n"
                "### Full Amortization Loop\n"
                "1. Generate synthetic data with SDG\n"
                "2. Fine-tune a small model on the generated data\n"
                "3. Replace expensive frontier model calls with your custom model\n\n"
                "Check the **Flows** page to see available SDG flow templates."
            ),
            suggested_action=SuggestedAction(
                type="view_flows",
                label="View Flows",
                config={},
            ),
        )

    def _handle_artifacts(self, message: str) -> AgentResponse:
        return AgentResponse(
            message=(
                "Job outputs are registered as **artifacts** when a job completes. "
                "For training jobs, artifacts include:\n\n"
                "- **Adapter weights** — LoRA adapter files\n"
                "- **Tokenizer** — Saved tokenizer configuration\n"
                "- **Training metrics** — Per-step loss and learning rate data\n\n"
                "For SDG jobs:\n"
                "- **Generated dataset** — The synthetic training data (JSONL)\n"
                "- **Checkpoints** — Intermediate generation state\n\n"
                "View artifacts on the **Artifacts** tab of any completed job."
            ),
            suggested_action=SuggestedAction(
                type="view_jobs",
                label="View Jobs",
                config={},
            ),
        )

    def _handle_help(self, message: str) -> AgentResponse:
        return AgentResponse(
            message=(
                "Hi! I'm the Amortized assistant. I can help you with:\n\n"
                "- **Fine-tuning models** — Set up LoRA SFT training jobs\n"
                "- **Generating training data** — Configure SDG pipelines\n"
                "- **Estimating VRAM** — Check GPU memory requirements\n"
                "- **Checking job status** — Monitor running jobs\n"
                "- **Understanding outputs** — Navigate job artifacts\n\n"
                "What would you like to do?"
            ),
        )

    def process_message(self, message: str) -> AgentResponse:
        """Match user message against intents and return a response."""
        for pattern, handler in self._intents:
            if pattern.search(message):
                return handler(message)

        return AgentResponse(
            message=(
                "I'm not sure I understand. I can help with:\n\n"
                "- **Fine-tuning** — \"I want to fine-tune a model\"\n"
                "- **Data generation** — \"Generate training data for X\"\n"
                "- **VRAM estimation** — \"How much VRAM do I need?\"\n"
                "- **Job status** — \"What's the status of my job?\"\n"
                "- **Available flows** — \"What flows are available?\"\n\n"
                "Try asking about one of these topics!"
            ),
        )


# Singleton agent instance
agent = AmortizedAgent()
