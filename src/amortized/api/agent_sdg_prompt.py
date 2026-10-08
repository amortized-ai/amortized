"""SDG prompt binding: the prompts the user approved in show_prompt previews must be the
ones that end up in the confirm card AND the launched job. A recipe carries more than one
prompt (an input/ticket generator and an assessor), so each approved prompt is bound back
to the recipe column it most closely matches — not "the last one shown" into the assessor
slot. With no preview the model's prompts are left untouched.

Called from `get_session_messages` in agent.py via `_bind_sdg_prompt_to_approved`.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from amortized.api.agent_parts import _get_tool_input, _tool_name

_SHOW_PROMPT_TOOL = "show_prompt"
_VALIDATE_SDG_TOOL = "validate_sdg_job"


def _shown_prompts(messages: list[dict[str, Any]]) -> list[str]:
    """Every prompt the user was shown via show_prompt in `messages` (chronological),
    de-duplicated preserving order — a driver may re-show a prompt before confirming."""
    out: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for part in msg.get("parts") or []:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool"
                and _tool_name(part) == _SHOW_PROMPT_TOOL
            ):
                shown = str(_get_tool_input(part).get("prompt") or "").strip()
                if shown and shown not in out:
                    out.append(shown)
    return out


def _rewrite_processor_system_messages(
    config: dict[str, Any], assessor_col: str, new_prompt: str
) -> bool:
    """Set the SFT processor template's `system` message to `new_prompt` for the
    processor whose assistant turn references the assessor column — the assessor prompt
    ships both on its column and in the processor's system turn. Returns True if it
    changed anything."""
    ref_pat = re.compile(r"\{\{\s*" + re.escape(assessor_col) + r"\s*\}\}")
    changed = False
    for proc in config.get("processors") or []:
        if not isinstance(proc, dict):
            continue
        template = proc.get("template")
        messages = template.get("messages") if isinstance(template, dict) else None
        if not isinstance(messages, list):
            continue
        refs_assessor = any(
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and ref_pat.search(m["content"])
            for m in messages
        )
        if not refs_assessor:
            continue
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            if msg.get("content") != new_prompt:
                msg["content"] = new_prompt
                changed = True
    return changed


def _bind_prompts_to_config(config: dict[str, Any], shown: list[str]) -> bool:
    """Overwrite each prompt-bearing recipe column's `system_prompt` with the approved
    preview it most closely matches (greedy one-to-one by text similarity), and align the
    SFT processor's system turn when the assessor column is rebound. Returns True if it
    changed anything."""
    from amortized.api.jobs import _assessor_column_name

    columns = config.get("columns")
    if not isinstance(columns, list) or not shown:
        return False
    prompt_cols = [
        c
        for c in columns
        if isinstance(c, dict)
        and isinstance(c.get("system_prompt"), str)
        and c["system_prompt"].strip()
    ]
    if not prompt_cols:
        return False
    assessor_col = _assessor_column_name(config)
    scored: list[tuple[float, int, int]] = []
    for si, s in enumerate(shown):
        for ci, col in enumerate(prompt_cols):
            ratio = difflib.SequenceMatcher(None, s, col["system_prompt"]).ratio()
            scored.append((ratio, si, ci))
    scored.sort(key=lambda t: t[0], reverse=True)
    used_s: set[int] = set()
    used_c: set[int] = set()
    changed = False
    for _ratio, si, ci in scored:
        if si in used_s or ci in used_c:
            continue
        used_s.add(si)
        used_c.add(ci)
        col = prompt_cols[ci]
        if col["system_prompt"] != shown[si]:
            col["system_prompt"] = shown[si]
            changed = True
        if (
            assessor_col is not None
            and col.get("name") == assessor_col
            and _rewrite_processor_system_messages(config, assessor_col, shown[si])
        ):
            changed = True
    return changed


def _bind_validated_sdg_prompts(data: dict[str, Any], shown: list[str]) -> bool:
    """Rewrite a ValidatedJobConfig dict ({config, assessor_prompt, prompts, ...}) so the
    config (what the job runs with) carries the approved prompts, then refresh the
    card's display fields (`prompts`, `assessor_prompt`) from the rebound config."""
    config = data.get("config")
    if not isinstance(config, dict) or not _bind_prompts_to_config(config, shown):
        return False
    from amortized.api.jobs import _recipe_prompts

    prompts = _recipe_prompts(config)
    data["prompts"] = [p.model_dump() for p in prompts]
    data["assessor_prompt"] = next((p.text for p in prompts if p.role == "assessor"), None)
    return True


def _bind_validate_part(part: dict[str, Any], shown: list[str]) -> None:
    """Bind the approved prompts into a validate_sdg_job tool part's OUTPUT (the
    ValidatedJobConfig the UI reads for the card and the create payload)."""
    containers: list[tuple[dict[str, Any], str]] = []
    state = part.get("state")
    if isinstance(state, dict) and "output" in state:
        containers.append((state, "output"))
    if "output" in part:
        containers.append((part, "output"))
    for obj, key in containers:
        raw = obj.get(key)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict) and _bind_validated_sdg_prompts(parsed, shown):
                obj[key] = json.dumps(parsed)
        elif isinstance(raw, dict):
            _bind_validated_sdg_prompts(raw, shown)


def _bind_sdg_prompt_to_approved(messages: list[dict[str, Any]]) -> None:
    """In-place: if the user was shown prompts (show_prompt) and these messages carry a
    validate_sdg_job confirm card, bind every approved prompt to the recipe column it
    matches. No preview → no-op. See the module docstring for rationale."""
    if not isinstance(messages, list):
        return
    shown = _shown_prompts(messages)
    if not shown:
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for part in msg.get("parts") or []:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool"
                and _tool_name(part) == _VALIDATE_SDG_TOOL
            ):
                _bind_validate_part(part, shown)
