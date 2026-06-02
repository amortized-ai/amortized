"""Tests for the agent intent matching and response generation."""

import pytest

from amortized_runtime.agent import AmortizedAgent


@pytest.fixture
def agent() -> AmortizedAgent:
    return AmortizedAgent()


class TestTrainingIntent:
    def test_fine_tune_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("I want to fine-tune a model")
        assert "LoRA SFT" in resp.message
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "create_training_job"

    def test_training_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("How do I train a model?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "create_training_job"

    def test_lora_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Set up LoRA for my project")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "create_training_job"

    def test_llama_model_detection(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Fine-tune Llama for classification")
        assert resp.suggested_action is not None
        assert "llama" in resp.suggested_action.config["model_path"].lower()

    def test_mistral_model_detection(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Train a Mistral model")
        assert resp.suggested_action is not None
        assert "mistral" in resp.suggested_action.config["model_path"].lower()

    def test_default_model_is_qwen(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("I want to fine-tune a model")
        assert resp.suggested_action is not None
        assert "Qwen" in resp.suggested_action.config["model_path"]


class TestSDGIntent:
    def test_generate_data_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Generate training data for classification")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "create_sdg_job"

    def test_synthetic_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("I need synthetic data")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "create_sdg_job"

    def test_sdg_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("How does SDG work?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "create_sdg_job"


class TestStatusIntent:
    def test_status_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("What's the status of my job?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "view_jobs"

    def test_check_progress(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Check on my running job")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "view_jobs"


class TestEstimationIntent:
    def test_vram_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("How much VRAM do I need?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "estimate_memory"

    def test_gpu_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Do I have enough GPU memory?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "estimate_memory"


class TestFlowsIntent:
    def test_flows_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("What flows are available?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "view_flows"

    def test_capabilities_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("What capabilities does Amortized have?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "view_flows"


class TestArtifactsIntent:
    def test_artifacts_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Where are my artifacts?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "view_jobs"

    def test_output_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("How do I download the output?")
        assert resp.suggested_action is not None
        assert resp.suggested_action.type == "view_jobs"


class TestHelpIntent:
    def test_help_keyword(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Help me get started")
        assert "fine-tuning" in resp.message.lower() or "Fine-tuning" in resp.message

    def test_hello(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("Hello!")
        assert resp.suggested_action is None


class TestFallback:
    def test_unknown_message(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("pizza recipe")
        assert "not sure" in resp.message.lower() or "I'm not sure" in resp.message
        assert resp.suggested_action is None

    def test_response_has_message(self, agent: AmortizedAgent) -> None:
        resp = agent.process_message("xyzzy")
        assert len(resp.message) > 0
