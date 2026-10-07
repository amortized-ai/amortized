"""The SDG assessor-prompt binding: the prompt the user approved in a show_prompt
preview must be the one that ends up in the confirm card AND the launched job. The proxy
copies the last previewed prompt into the validate_sdg_job output when a preview happened;
with no preview it leaves the model's prompt untouched."""

from __future__ import annotations

import json
from typing import Any

from amortized.api import agent


def _tool_part(
    tool: str, *, input: dict[str, Any] | None = None, output: Any = None
) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "tool", "tool": f"mcp_amortized__{tool}"}
    if input is not None:
        part["input"] = input
    if output is not None:
        part["state"] = {"output": output}
    return part


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _config(*, input_prompt: str, output_prompt: str, processor_system: str) -> dict[str, Any]:
    """Mirror the task-distillation reference payload: an input-generator column, an
    output column whose system_prompt is the assessor prompt, and a processor whose
    assistant turn emits {{ output }} and whose system message repeats that prompt."""
    return {
        "columns": [
            {"name": "generated_input", "system_prompt": input_prompt},
            {"name": "output", "system_prompt": output_prompt},
        ],
        "processors": [
            {
                "template": {
                    "messages": [
                        {"role": "system", "content": processor_system},
                        {"role": "user", "content": "{{ generated_input }}"},
                        {"role": "assistant", "content": "{{ output }}"},
                    ]
                }
            }
        ],
    }


def _validated(cfg: dict[str, Any], assessor_prompt: str) -> dict[str, Any]:
    return {
        "job_type": "sdg",
        "config": cfg,
        "parent_job_id": "",
        "assessor_prompt": assessor_prompt,
    }


class TestRewriteAssessorPrompt:
    def test_rewrites_output_column_and_processor_system_only(self) -> None:
        cfg = _config(input_prompt="PERSONA", output_prompt="DRIFT", processor_system="DRIFT_SYS")
        assert agent._rewrite_assessor_prompt(cfg, "APPROVED") is True
        assert cfg["columns"][1]["system_prompt"] == "APPROVED"  # output column
        assert cfg["processors"][0]["template"]["messages"][0]["content"] == "APPROVED"
        # The input-generator column's own prompt is a DIFFERENT prompt — never touched.
        assert cfg["columns"][0]["system_prompt"] == "PERSONA"

    def test_no_processors_returns_false(self) -> None:
        assert agent._rewrite_assessor_prompt({"columns": []}, "X") is False
        assert agent._rewrite_assessor_prompt({}, "X") is False


class TestLastShownPrompt:
    def test_returns_latest_and_strips(self) -> None:
        messages = [
            {"parts": [_tool_part("show_prompt", input={"prompt": " FIRST "})]},
            {"parts": [_tool_part("show_prompt", input={"prompt": "LATEST"})]},
        ]
        assert agent._last_shown_prompt(messages) == "LATEST"

    def test_single_is_stripped(self) -> None:
        messages = [{"parts": [_tool_part("show_prompt", input={"prompt": "  ONLY  "})]}]
        assert agent._last_shown_prompt(messages) == "ONLY"

    def test_none_when_never_shown(self) -> None:
        assert agent._last_shown_prompt([{"parts": [_text_part("hello")]}]) is None


class TestBindSdgPromptToApproved:
    def _messages(
        self, *, shown: str | None, output: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        validate = _tool_part("validate_sdg_job", output=output)
        messages: list[dict[str, Any]] = []
        if shown is not None:
            messages.append({"parts": [_tool_part("show_prompt", input={"prompt": shown})]})
        messages.append({"parts": [validate]})
        return messages, validate

    def test_copies_approved_into_card_and_config_when_output_is_json_string(self) -> None:
        cfg = _config(input_prompt="PERSONA", output_prompt="DRIFT", processor_system="DRIFT")
        messages, validate = self._messages(
            shown="APPROVED", output=json.dumps(_validated(cfg, "DRIFT"))
        )
        agent._bind_sdg_prompt_to_approved(messages)
        out = json.loads(validate["state"]["output"])
        assert out["assessor_prompt"] == "APPROVED"  # the card display
        assert out["config"]["columns"][1]["system_prompt"] == "APPROVED"  # the launched job
        assert out["config"]["processors"][0]["template"]["messages"][0]["content"] == "APPROVED"
        assert out["config"]["columns"][0]["system_prompt"] == "PERSONA"  # input persona untouched

    def test_copies_when_output_is_dict(self) -> None:
        cfg = _config(input_prompt="PERSONA", output_prompt="DRIFT", processor_system="DRIFT")
        messages, validate = self._messages(shown="APPROVED", output=_validated(cfg, "DRIFT"))
        agent._bind_sdg_prompt_to_approved(messages)
        out = validate["state"]["output"]
        assert out["assessor_prompt"] == "APPROVED"
        assert out["config"]["columns"][1]["system_prompt"] == "APPROVED"

    def test_noop_without_preview(self) -> None:
        # Knowledge-ingestion / classification generate the prompt without a preview —
        # there is nothing the user approved to bind against, so leave it alone.
        cfg = _config(
            input_prompt="PERSONA", output_prompt="GENERATED", processor_system="GENERATED"
        )
        messages, validate = self._messages(
            shown=None, output=json.dumps(_validated(cfg, "GENERATED"))
        )
        agent._bind_sdg_prompt_to_approved(messages)
        out = json.loads(validate["state"]["output"])
        assert out["assessor_prompt"] == "GENERATED"
        assert out["config"]["columns"][1]["system_prompt"] == "GENERATED"

    def test_ignores_non_sdg_validate(self) -> None:
        training = _tool_part(
            "validate_training_job", output=json.dumps({"config": {"model": "x"}})
        )
        messages = [
            {"parts": [_tool_part("show_prompt", input={"prompt": "APPROVED"})]},
            {"parts": [training]},
        ]
        agent._bind_sdg_prompt_to_approved(messages)
        assert json.loads(training["state"]["output"]) == {"config": {"model": "x"}}
