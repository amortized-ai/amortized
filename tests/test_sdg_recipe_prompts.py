"""Resolving the reviewable prompts of an SDG recipe for the confirmation card.

A recipe commonly carries more than one system prompt — an input/ticket generator and
an assessor — and the card must surface ALL of them (the user reported "the prompts only
show one"). `_recipe_prompts` returns every prompt-bearing column tagged with its role,
input generators first and the assessor (the SFT assistant target) last.
"""

from __future__ import annotations

from typing import Any

from amortized.api import jobs


def _config(**cols: str) -> dict[str, Any]:
    """Columns keyed name->system_prompt; `output` is the SFT assistant target."""
    return {
        "columns": [{"name": name, "system_prompt": prompt} for name, prompt in cols.items()],
        "processors": [
            {
                "template": {
                    "messages": [
                        {"role": "system", "content": cols.get("output", "")},
                        {"role": "user", "content": "{{ generated_input }}"},
                        {"role": "assistant", "content": "{{ output }}"},
                    ]
                }
            }
        ],
    }


def test_resolves_both_input_and_assessor_prompts() -> None:
    cfg = _config(generated_input="TICKET", output="ASSESS")
    prompts = jobs._recipe_prompts(cfg)
    assert [(p.role, p.column, p.text) for p in prompts] == [
        ("input", "generated_input", "TICKET"),  # input generators first
        ("assessor", "output", "ASSESS"),  # SFT target last
    ]
    assert prompts[0].label == "Generated input prompt"
    assert prompts[1].label == "Assessor system prompt"


def test_assessor_prompt_mirrors_the_assessor_role_entry() -> None:
    cfg = _config(generated_input="TICKET", output="ASSESS")
    assert jobs._assessor_prompt(cfg) == "ASSESS"


def test_multiple_input_generators_all_surface() -> None:
    cfg = _config(persona="PERSONA", generated_input="TICKET", output="ASSESS")
    prompts = jobs._recipe_prompts(cfg)
    assert [p.role for p in prompts] == ["input", "input", "assessor"]
    assert {p.text for p in prompts if p.role == "input"} == {"PERSONA", "TICKET"}


def test_no_identifiable_assessor_yields_only_inputs() -> None:
    # No SFT processor -> the assessor column can't be identified; input prompts still show.
    cfg: dict[str, Any] = {"columns": [{"name": "generated_input", "system_prompt": "TICKET"}]}
    prompts = jobs._recipe_prompts(cfg)
    assert [(p.role, p.text) for p in prompts] == [("input", "TICKET")]
    assert jobs._assessor_prompt(cfg) is None


def test_columns_without_prompts_are_ignored() -> None:
    cfg = {
        "columns": [
            {"name": "sampler", "column_type": "category"},  # no system_prompt
            {"name": "output", "system_prompt": "ASSESS"},
        ],
        "processors": _config(output="ASSESS")["processors"],
    }
    prompts = jobs._recipe_prompts(cfg)
    assert [(p.role, p.text) for p in prompts] == [("assessor", "ASSESS")]
