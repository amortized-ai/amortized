# Design: a config-mirror module (`core/mirror.py`)

**Status:** proposal (spec only — no code yet)
**Branch:** `feat/mirror-module` (off the `#496→#497→#498` stack tip)

## Problem

"Derive a new job config from an existing one by carrying the parts that must stay
identical and changing only what's meant to change" is implemented **three times**
today, each with its own ad-hoc notion of *what the recipe is*:

| Site | What it does | Its "don't carry" set |
|---|---|---|
| `clone_sdg_config_for_eval` (jobs.py:733) | build an eval-set SDG from a training SDG | `non_recipe_keys = {num_records, parent_job_id, mode}` |
| `retry_job` (jobs.py:512) | re-submit a failed/cancelled eval | `_RETRY_STRIP_KEYS = (eval_data_path, port, served_model_name)` + `parent_job_id` |
| preview→full gate (`_sdg_config_signature`, agent_confirm.py:61; `_SDG_NON_RECIPE_FIELDS`) | gate that a full SDG job matches its approved preview | `_SDG_NON_RECIPE_FIELDS = {mode, num_records, parent_job_id}` |

Plus a **comparator/reporter** pair that is the inverse of the same idea — detect when a
mirror has *drifted*: `_sdg_signature` (jobs.py:988, a 5-tuple of recipe fields) and
`_training_sdg_mirror_warning` (jobs.py:1028, a component-by-component diff).

Three drop-sets and two signature functions, none sharing a source. Two concrete
consequences:

1. **Drift bug.** `clone_sdg_config_for_eval` copies the recipe *by exclusion* (denylist),
   but `_sdg_signature` compares it *by inclusion* (a fixed 5-tuple). Add a new
   recipe-defining field and the clone carries it while the mirror check ignores it — a
   divergence in that field is silently never warned. The producer and the verifier
   disagree on what "the recipe" is.
2. **Arbitrary restriction.** `retry_job` hard-rejects everything but eval
   (jobs.py:522), although `request_config`/`retry_of` (migration 0003) are type-agnostic.
   SDG and training can't be re-run at all, and no path exists for "re-run but change one
   knob."

## Core idea: one field classification, shared by clone + signature

Every mirror operation is `source config → derived config` under a **field policy**. The
honest primitive is a classification of config keys into three roles, from which both the
clone's drop-set *and* the signature's carry-set derive — so they can never disagree:

- **`RUNTIME_KEYS`** — worker-injected at dispatch, always recomputed by the config
  builder. Stripped on *every* mirror (this is what `_RETRY_STRIP_KEYS` is, generalized).
  Never part of the signature.
- **`SEMANTIC_KEYS`** (per job type) — sample-size / lineage knobs the caller is *expected*
  to set: SDG `{num_records, mode}`, training `{num_train_epochs, ...}`, eval `{...}`.
  `parent_job_id`/`training_job_id` are lineage and live here too. Carried verbatim
  unless the caller passes an override. Never part of the signature.
- **recipe (everything else)** — the fields that define the task and must stay identical.
  The signature is computed over *exactly this set* (`all_keys − RUNTIME_KEYS − SEMANTIC_KEYS`).

The drift bug disappears by construction: a new recipe field is, by default, in neither
excluded set, so it is both carried by the clone and folded into the signature.

## Proposed module `src/amortized/core/mirror.py` (pure — no FastAPI/opencode deps)

```python
@dataclass(frozen=True)
class MirrorSpec:
    runtime_keys: frozenset[str]        # always stripped (recomputed by the builder)
    semantic_keys: frozenset[str]       # carried, but overridable by the caller

def clone(src: dict, spec: MirrorSpec, overrides: dict | None = None) -> dict:
    """Carry the recipe + semantic fields, strip runtime keys, apply overrides."""

def recipe_signature(cfg: dict, spec: MirrorSpec) -> Signature | None:
    """Stable, order-insensitive signature over recipe fields (= all − runtime − semantic).
    None when empty/unparseable (preserves current _sdg_signature semantics)."""

def diff_recipe(a: dict, b: dict, spec: MirrorSpec) -> list[str]:
    """Human-readable, component-by-component divergence (powers the mirror warning)."""
```

Specs are declared per job type in one place (the single source of truth that replaces the
three ad-hoc sets):

```python
SDG_MIRROR      = MirrorSpec(runtime_keys={...}, semantic_keys={"num_records", "mode", "parent_job_id"})
TRAINING_MIRROR = MirrorSpec(runtime_keys={...}, semantic_keys={"num_train_epochs", ..., "parent_job_id", "data_run_id"})
EVAL_MIRROR     = MirrorSpec(runtime_keys={"eval_data_path", "port", "served_model_name"},
                             semantic_keys={"parent_job_id", "training_job_id", "eval_data_run_id"})
```

**Hard boundary:** `core/mirror.py` is pure. The API route module (`jobs.py`) and the proxy
(`agent_confirm.py`) both *import* it; neither imports the other. The proxy keeps only its
tool-call-part→config adapter locally and reuses `recipe_signature`.

## How the six use cases map onto it

| # | Use case | Covered by | Notes |
|---|---|---|---|
| 1 | **Generalized retry/rerun** (SDG, training, eval) | `clone(src, spec)` from the `request_config` snapshot | lift the eval-only restriction; always via a confirm card (see guardrails) |
| 2 | **Rerun with a tweak** ("train again, 5 epochs" / "regenerate, 2000 records") | `clone(src, spec, overrides={...})` | overrides are exactly the `semantic_keys` the caller sets; shown in the confirm card |
| 3 | **Preview → full-run** | `recipe_signature` (replaces `_sdg_config_signature`) | one signature fn for gate + warning; `mode` is semantic so preview≈full |
| 4 | **Split siblings** (`split_run_id` / `complement_run_id`) | `recipe_signature` equality across the two runs' source configs | expresses "same recipe, different partition" as data, not prose |
| 5 | **Cross-model eval on the same eval set** | `clone(eval_cfg, EVAL_MIRROR, overrides={"training_job_id": B})` | mirror an eval, swap only the model under test |
| 6 | **Generalized drift warning** | `diff_recipe` (generalizes `_training_sdg_mirror_warning`) | any declared-source relationship can warn on divergence |

All six are in scope. 1–3 collapse live duplication and unlock real features; 4–6 reuse the
same two functions on new call sites.

## Guardrails that must NOT be folded into the module (policy ≠ mechanics)

The module unifies the *plumbing*. The *decisions* stay at the call sites:

- **Every rerun routes through a confirm card — for all types, identical or with
  overrides.** Rerun is allowed for every type including training (confirmed); but because
  any rerun can burn real GPU, none of them dispatches silently. The module just hands back
  a config (`clone(...)`); the call site always renders a `validate_*` card the user
  Confirms before `create_*`. There is no one-click retry.
  - **Behavior change:** this *replaces* today's one-click `retry_job` endpoint
    (jobs.py:512), which re-creates an eval directly with no card. Under this rule even an
    identical eval retry shows a confirm card first. The `/{job_id}/retry` endpoint either
    becomes a thin "pre-fill the config, then present the card" step or is dropped in favor
    of the normal `validate_eval_job` flow seeded from the snapshot.
  - Overrides vs identical no longer branch the *path* (both confirm); overrides only
    change *what the card shows* (the diff the user reviews).
- **`retry_of` lineage** continues to be set by the create path, not the module.
- **Stage-gate dependency invariant** (upstream job must be `succeeded`) is unchanged — a
  rerun of a downstream job still can't advance past a non-succeeded upstream.

## Call sites to rewire

1. `clone_sdg_config_for_eval` → thin HTTP shell over `clone(parent_cfg, SDG_MIRROR, overrides={"num_records": n})`.
2. `retry_job` → generalize: drop the eval-only check; `clone(snapshot, SPEC_FOR[type])`;
   route **all** reruns (every type, identical or overridden) through the confirm card +
   stage gate — no direct one-click re-create.
3. `_sdg_signature` / `_training_sdg_mirror_warning` → call `recipe_signature` / `diff_recipe`.
4. `_sdg_config_signature` (proxy preview gate) → call `recipe_signature` (keep the local part adapter).

## Tests

- `core/mirror`: drop-set and signature carry-set derive from one classification (add a
  fake recipe field → it is both cloned and signed); `clone` honors overrides; signature is
  order-insensitive; `None` on empty.
- Regression parity: `clone_sdg_config_for_eval`, `retry_job`, the preview gate, and the
  mirror warning produce the **same** results as today on existing fixtures (pure refactor).
- New behavior: SDG/training retry; rerun-with-override; cross-model eval mirror; a training
  rerun still requires a confirm card.

## Sequencing

- Land **on top of the merged `#496→#497→#498` stack** (it depends on their jobs.py /
  agent_confirm.py changes), as its own PR — do not retrofit into the four green PRs.
- Phase A (high value, low risk): module + rewire cases 1–3 (parity refactor + retry
  generalization). Phase B: cases 4–6 on new call sites.

## Non-goals

- No change to the config *builder* (worker-side resolution of runtime keys).
- No change to the dependency/stage-gate invariant.
- No merging of the per-record **overlap/leakage** signatures (`_input_signatures`,
  `_eval_overlap_warning`) — different question (row content, not recipe), stays separate.
