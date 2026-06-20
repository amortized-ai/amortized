# Judge Templates Migration: asynth → amortized

Move all judge YAML templates out of the asynth package into amortized. asynth becomes a pure Python library with zero bundled config files — just like TRL provides trainer classes but no YAML recipes.

## Rationale

asynth is a backend engine. No one uses it directly. The only consumer is the amortized control plane. User-facing configs should live where users interact with them — in amortized.

Current inconsistency:
- SDG configs → amortized `recipes/sdg/` ✓
- Training configs → amortized `recipes/training/` ✓
- Serve configs → amortized `recipes/serve/` ✓
- Judge templates → **asynth** `src/asynth/judges/templates/` ✗

After migration, all YAML configs live in amortized. asynth is pure Python.

---

## What Moves

### Files to move from asynth to amortized

Source: `/Users/shiv/workspace/asynth/src/asynth/judges/templates/`

```
asynth/src/asynth/judges/templates/
  generic/
    safety.yaml
    instruction_following.yaml
    truthfulness.yaml
    topic_adherence.yaml
    format_compliance.yaml
  doc_qa/
    groundedness.yaml
    completeness.yaml
    relevance.yaml
  code/
    code_quality.yaml
    correctness.yaml
    maintainability.yaml
    performance.yaml
    security.yaml
  rule_based/
    regex_match_phone.yaml
    regex_no_error_keywords.yaml
```

Destination in amortized: `recipes/judges/`

```
amortized/recipes/judges/
  generic/
    safety.yaml
    instruction_following.yaml
    truthfulness.yaml
    topic_adherence.yaml
    format_compliance.yaml
  doc_qa/
    groundedness.yaml
    completeness.yaml
    relevance.yaml
  code/
    code_quality.yaml
    correctness.yaml
    maintainability.yaml
    performance.yaml
    security.yaml
  rule_based/
    regex_match_phone.yaml
    regex_no_error_keywords.yaml
```

The YAML content does NOT change. Copy the files as-is.

---

## Changes to asynth

### 1. Delete `src/asynth/judges/templates/` directory

Remove the entire `templates/` directory and all 16 YAML files.

### 2. Remove `load_template()` and `list_templates()` from `src/asynth/judges/__init__.py`

These functions load YAML from the bundled templates directory. With no templates, they have nothing to load.

Delete:
- `load_template(name: str) -> JudgeConfig` — loads a YAML template by name
- `list_templates() -> list[str]` — lists available template names

### 3. Remove template-related imports from `src/asynth/__init__.py`

Remove `load_template` and `list_templates` from the top-level re-exports. The public API becomes:

```python
# src/asynth/__init__.py — after migration
from asynth.judges import (
    SimpleJudge,
    RuleBasedJudge,
    JudgeConfig,
    JudgeOutput,
    JudgeOutputField,
    BaseJudge,
    BaseRule,
    RegexRule,
    create_judge,
)
from asynth.judges.base_judge import judge  # convenience function — keep or remove, see below
```

### 4. Update or remove the `judge()` convenience function

The `judge()` function in `src/asynth/judges/__init__.py` currently does:
```python
def judge(template, data, model, **model_kwargs):
    config = load_template(template)  # ← this breaks
    j = create_judge(config, inference_config)
    return j.judge(data)
```

Two options:

**Option A (recommended): Remove `judge()` entirely.** The amortized server will load templates itself and call `create_judge()` + `.judge()` directly. The convenience function was only useful when templates were bundled.

**Option B: Refactor `judge()` to accept a `JudgeConfig` directly instead of a template name.** But then it's just a thin wrapper around `create_judge()` + `.judge()` with no real value.

Go with Option A.

### 5. Remove template-related test files

If there are tests in asynth that test `load_template()` or `list_templates()`, remove them. Tests for `SimpleJudge`, `RuleBasedJudge`, `create_judge()` should remain — they take `JudgeConfig` objects directly, not template names.

### 6. Update `pyproject.toml`

If asynth's `pyproject.toml` has `[tool.setuptools.package-data]` entries that include `templates/**/*.yaml`, remove them. No more bundled data files.

Check for entries like:
```toml
[tool.setuptools.package-data]
asynth = ["judges/templates/**/*.yaml"]
```

Remove this section.

---

## Changes to amortized

### 1. Create `recipes/judges/` directory

Copy all 16 YAML files from asynth into `recipes/judges/` preserving the subdirectory structure (generic/, doc_qa/, code/, rule_based/).

### 2. Add template loading to amortized server

The amortized server needs a way to load judge templates by name. Add a utility function, likely in the evaluators module:

```python
# server/src/amortized/utils/judges.py (or similar)

import yaml
from pathlib import Path
from asynth.judges import JudgeConfig, create_judge

RECIPES_DIR = Path(__file__).parent.parent.parent.parent / "recipes"
JUDGES_DIR = RECIPES_DIR / "judges"

def load_judge_template(name: str) -> dict:
    """Load a judge template YAML by name.
    
    Args:
        name: Template name like "generic/safety" or "code/correctness"
    
    Returns:
        Parsed YAML dict suitable for JudgeConfig construction
    """
    path = JUDGES_DIR / f"{name}.yaml"
    if not path.exists():
        available = list_judge_templates()
        raise FileNotFoundError(
            f"Judge template '{name}' not found. Available: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)

def list_judge_templates() -> list[str]:
    """List available judge template names."""
    templates = []
    for path in sorted(JUDGES_DIR.rglob("*.yaml")):
        name = str(path.relative_to(JUDGES_DIR)).removesuffix(".yaml")
        templates.append(name)
    return templates
```

The exact location and implementation depends on amortized's project structure. The key point: template loading moves from asynth to amortized.

### 3. Update evaluators API

If the evaluators API (`server/src/amortized/api/evaluators.py`) references asynth's `load_template()`, update it to use the new amortized-local loader.

### 4. Update eval runner

If the eval container or runner references `load_template()`, update it similarly. The eval container needs access to the `recipes/judges/` directory — either mount it as a volume or copy the templates into the container at build time.

Check: `containers/eval/` or any eval-related Dockerfile. If the eval container uses asynth's bundled templates, it needs updating.

### 5. Update the synth container

Check if `containers/synth/` uses judge templates (e.g., for inline quality checks during synthesis). If so, mount or copy `recipes/judges/` into that container too.

---

## Migration Order

Do this in sequence — asynth changes first, then amortized.

### Step 1: amortized — add templates (non-breaking)

1. Create `recipes/judges/` directory in amortized
2. Copy all 16 YAML files from asynth
3. Add template loading utility to amortized server
4. Update any amortized code that calls `asynth.load_template()` to use the new local loader
5. Test that the evaluators API still works

### Step 2: asynth — remove templates (breaking, but no external consumers)

1. Delete `src/asynth/judges/templates/` directory
2. Remove `load_template()`, `list_templates()`, `judge()` from `judges/__init__.py`
3. Remove re-exports from `src/asynth/__init__.py`
4. Remove `package-data` entries from `pyproject.toml`
5. Remove related tests
6. Bump version (0.1.2 or 0.2.0)

### Step 3: amortized — update asynth dependency

1. Update `asynth>=0.1.2` (or whatever the new version is) in amortized's dependencies
2. Verify everything works end to end

---

## What asynth Keeps (unchanged)

All Python code stays in asynth:

| Module | What it provides |
|---|---|
| `judges/base_judge.py` | `BaseJudge`, `JudgeOutput`, `JudgeOutputField` |
| `judges/simple_judge.py` | `SimpleJudge` (LLM-based judge) |
| `judges/rule_based_judge.py` | `RuleBasedJudge` (deterministic judge) |
| `judges/rules/base_rule.py` | `BaseRule` ABC |
| `judges/rules/regex.py` | `RegexRule` implementation |
| `judges/__init__.py` | `create_judge()` factory (takes JudgeConfig, returns SimpleJudge or RuleBasedJudge) |
| `configs/judge_config.py` | `JudgeConfig`, `JudgeParams`, `RuleJudgeParams` dataclasses |

The judge engine is asynth's. The judge configs are amortized's. Clean separation.

## What amortized Gains

```
amortized/recipes/judges/
  generic/
    safety.yaml                    # harmful content detection
    instruction_following.yaml     # did it follow instructions?
    truthfulness.yaml              # factual accuracy
    topic_adherence.yaml           # stayed on topic?
    format_compliance.yaml         # correct output format?
  doc_qa/
    groundedness.yaml              # answer grounded in context?
    completeness.yaml              # answer covers all key points?
    relevance.yaml                 # answer relevant to question?
  code/
    code_quality.yaml              # readability, structure, DRY
    correctness.yaml               # functional correctness
    maintainability.yaml           # modularity, testability
    performance.yaml               # time/space complexity
    security.yaml                  # input validation, injection
  rule_based/
    regex_match_phone.yaml         # regex: phone number pattern
    regex_no_error_keywords.yaml   # regex: no error/fail/exception
```

---

## Verification

After migration is complete:

1. `pip install asynth` installs no YAML files — verify with:
   ```bash
   python -c "import asynth; print(asynth.__file__)"
   # Check that directory has no templates/ subdirectory
   ```

2. `from asynth import load_template` raises `ImportError` — it no longer exists

3. `from asynth import SimpleJudge, create_judge, JudgeConfig` still works

4. amortized's `list_judge_templates()` returns all 16 template names

5. amortized's eval pipeline can load a judge template and run it:
   ```python
   config = load_judge_template("generic/safety")
   judge = create_judge(JudgeConfig(**config), inference_config)
   results = judge.judge(test_data)
   ```

6. All existing eval recipes (`examples/ticket-classifier/eval.yaml`, etc.) still work
