"""Read the skill-level action contract from opencode SKILL.md frontmatter.

Each `agents/*/skills/**/SKILL.md` carries YAML frontmatter with `name` and a
`metadata:` block; the native-skills migration lets us declare, per skill, that it
must culminate in a tool emission:

    metadata:
      stage: sdg
      action_required: true
      emits: [validate_sdg_job]

The monitor reads this as the authored contract: which tool each sub-skill is
expected to emit. The stall detector uses the union of `emits` as the set of
"deliverable" tools (what counts as real progress vs talk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_AGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "agents"


def _parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        data = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def load_skill_contracts(agents_dir: Path = _AGENTS_DIR) -> dict[str, dict[str, Any]]:
    """Map skill name -> {action_required: bool, emits: [tool], stage: str} parsed
    from every SKILL.md's frontmatter metadata. Empty if the agents tree is absent."""
    contracts: dict[str, dict[str, Any]] = {}
    if not agents_dir.is_dir():
        return contracts
    for path in sorted(agents_dir.rglob("SKILL.md")):
        fm = _parse_frontmatter(path.read_text(encoding="utf-8"))
        name = fm.get("name")
        if not name:
            continue
        meta = fm.get("metadata") or {}
        emits = meta.get("emits") or []
        if isinstance(emits, str):
            emits = [emits]
        contracts[str(name)] = {
            "action_required": bool(meta.get("action_required")),
            "emits": [str(e) for e in emits],
            "stage": meta.get("stage"),
        }
    return contracts


def deliverable_tools(agents_dir: Path = _AGENTS_DIR) -> set[str]:
    """Union of all skills' `emits` tools — the authored set of deliverable/progress
    tools. Falls back to the known validate_* cards if no contracts are found."""
    tools: set[str] = set()
    for c in load_skill_contracts(agents_dir).values():
        tools.update(c["emits"])
    return tools or {"validate_sdg_job", "validate_training_job", "validate_eval_job"}
