"""The SDG prompt binding: the prompts the user approved in show_prompt previews must be
the ones that end up in the confirm card AND the launched job. A recipe carries more than
one prompt (an input/ticket generator and an assessor), so each approved prompt is bound
back to the recipe column it most closely matches — not "the last one shown" into the
assessor slot. With no preview the model's prompts are left untouched."""

from __future__ import annotations

import json
from typing import Any

from amortized.api import agent

# Realistic-ish prompts: each approved variant is a light paraphrase of its own config
# column and clearly unlike the other column, so similarity matching binds them correctly.
INPUT_CFG = "Write a realistic customer support ticket describing a billing problem."
INPUT_APPROVED = "Write a realistic customer support ticket about a billing problem the user hit."
ASSESS_CFG = "You are an assessor. Read the ticket and categorize it by urgency."
ASSESS_APPROVED = "You are a strict assessor. Read the ticket and categorize it by urgency level."


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


class TestShownPrompts:
    def test_returns_all_in_order_deduped_and_stripped(self) -> None:
        messages = [
            {"parts": [_tool_part("show_prompt", input={"prompt": " FIRST "})]},
            {"parts": [_tool_part("show_prompt", input={"prompt": "SECOND"})]},
            {"parts": [_tool_part("show_prompt", input={"prompt": "FIRST"})]},  # re-show
        ]
        assert agent._shown_prompts(messages) == ["FIRST", "SECOND"]

    def test_empty_when_never_shown(self) -> None:
        assert agent._shown_prompts([{"parts": [_text_part("hello")]}]) == []


class TestBindPromptsToConfig:
    def test_single_approved_binds_assessor_column_and_processor_only(self) -> None:
        cfg = _config(
            input_prompt=INPUT_CFG, output_prompt=ASSESS_CFG, processor_system=ASSESS_CFG
        )
        assert agent._bind_prompts_to_config(cfg, [ASSESS_APPROVED]) is True
        assert cfg["columns"][1]["system_prompt"] == ASSESS_APPROVED  # output/assessor column
        assert cfg["processors"][0]["template"]["messages"][0]["content"] == ASSESS_APPROVED
        # The input-generator column's own prompt is a DIFFERENT prompt — never touched.
        assert cfg["columns"][0]["system_prompt"] == INPUT_CFG

    def test_two_prompts_each_bind_their_own_column_regardless_of_order(self) -> None:
        cfg = _config(
            input_prompt=INPUT_CFG, output_prompt=ASSESS_CFG, processor_system=ASSESS_CFG
        )
        # Shown assessor FIRST, ticket LAST — the old "last shown -> assessor" rule would
        # have corrupted the assessor with the ticket prompt. Similarity binds correctly.
        assert agent._bind_prompts_to_config(cfg, [ASSESS_APPROVED, INPUT_APPROVED]) is True
        assert cfg["columns"][0]["system_prompt"] == INPUT_APPROVED  # input/ticket column
        assert cfg["columns"][1]["system_prompt"] == ASSESS_APPROVED  # assessor column
        assert cfg["processors"][0]["template"]["messages"][0]["content"] == ASSESS_APPROVED

    def test_no_prompt_columns_returns_false(self) -> None:
        assert agent._bind_prompts_to_config({"columns": []}, ["X"]) is False
        assert agent._bind_prompts_to_config({}, ["X"]) is False
        assert agent._bind_prompts_to_config(_config(
            input_prompt=INPUT_CFG, output_prompt=ASSESS_CFG, processor_system=ASSESS_CFG
        ), []) is False


class TestBindSdgPromptToApproved:
    def _messages(
        self, *, shown: list[str], output: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        validate = _tool_part("validate_sdg_job", output=output)
        messages: list[dict[str, Any]] = [
            {"parts": [_tool_part("show_prompt", input={"prompt": s})]} for s in shown
        ]
        messages.append({"parts": [validate]})
        return messages, validate

    def test_binds_single_approved_and_refreshes_display_fields(self) -> None:
        cfg = _config(input_prompt=INPUT_CFG, output_prompt=ASSESS_CFG, processor_system=ASSESS_CFG)
        messages, validate = self._messages(
            shown=[ASSESS_APPROVED], output=json.dumps(_validated(cfg, ASSESS_CFG))
        )
        agent._bind_sdg_prompt_to_approved(messages)
        out = json.loads(validate["state"]["output"])
        cols = out["config"]["columns"]
        proc_sys = out["config"]["processors"][0]["template"]["messages"][0]["content"]
        assert out["assessor_prompt"] == ASSESS_APPROVED  # the card display, refreshed
        assert cols[1]["system_prompt"] == ASSESS_APPROVED  # launched job
        assert proc_sys == ASSESS_APPROVED
        assert cols[0]["system_prompt"] == INPUT_CFG  # input untouched
        # The refreshed prompts list carries both roles.
        roles = {p["role"]: p["text"] for p in out["prompts"]}
        assert roles["assessor"] == ASSESS_APPROVED
        assert roles["input"] == INPUT_CFG

    def test_binds_two_prompts_to_their_columns(self) -> None:
        cfg = _config(input_prompt=INPUT_CFG, output_prompt=ASSESS_CFG, processor_system=ASSESS_CFG)
        messages, validate = self._messages(
            shown=[INPUT_APPROVED, ASSESS_APPROVED],
            output=json.dumps(_validated(cfg, ASSESS_CFG)),
        )
        agent._bind_sdg_prompt_to_approved(messages)
        out = json.loads(validate["state"]["output"])
        assert out["config"]["columns"][0]["system_prompt"] == INPUT_APPROVED
        assert out["config"]["columns"][1]["system_prompt"] == ASSESS_APPROVED
        roles = {p["role"]: p["text"] for p in out["prompts"]}
        assert roles == {"input": INPUT_APPROVED, "assessor": ASSESS_APPROVED}

    def test_binds_when_output_is_dict(self) -> None:
        cfg = _config(input_prompt=INPUT_CFG, output_prompt=ASSESS_CFG, processor_system=ASSESS_CFG)
        messages, validate = self._messages(
            shown=[ASSESS_APPROVED], output=_validated(cfg, ASSESS_CFG)
        )
        agent._bind_sdg_prompt_to_approved(messages)
        out = validate["state"]["output"]
        assert out["assessor_prompt"] == ASSESS_APPROVED
        assert out["config"]["columns"][1]["system_prompt"] == ASSESS_APPROVED

    def test_noop_without_preview(self) -> None:
        # Knowledge-ingestion / classification generate the prompt without a preview —
        # there is nothing the user approved to bind against, so leave it alone.
        cfg = _config(
            input_prompt=INPUT_CFG, output_prompt=ASSESS_CFG, processor_system=ASSESS_CFG
        )
        messages, validate = self._messages(
            shown=[], output=json.dumps(_validated(cfg, ASSESS_CFG))
        )
        agent._bind_sdg_prompt_to_approved(messages)
        out = json.loads(validate["state"]["output"])
        assert out["assessor_prompt"] == ASSESS_CFG
        assert out["config"]["columns"][1]["system_prompt"] == ASSESS_CFG

    def test_ignores_non_sdg_validate(self) -> None:
        training = _tool_part(
            "validate_training_job", output=json.dumps({"config": {"model": "x"}})
        )
        messages = [
            {"parts": [_tool_part("show_prompt", input={"prompt": ASSESS_APPROVED})]},
            {"parts": [training]},
        ]
        agent._bind_sdg_prompt_to_approved(messages)
        assert json.loads(training["state"]["output"]) == {"config": {"model": "x"}}
