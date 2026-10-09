# Design: a config-mirror module (`core/mirror.py`)

**Status:** implemented (Phase A + generalized retry/rerun). `core/mirror.py` + call-site
rewiring + confirm-on-rerun UI. Cases 4–6 (split siblings, cross-model eval mirror,
generalized drift warning) remain follow-ups.
**Branch:** `feat/mirror-module` (off the `#496→#497→#498` stack tip)

### Correction made during implementation

Reading the actual code showed the two SDG signatures have **intentionally different
strictness** and must stay separate (the original doc's "one `recipe_signature` replaces
both" was wrong):

- **Preview→full gate** (`_sdg_config_signature`) needs a **blanket** hash — any recipe
  edit must re-gate. This became `mirror.strict_signature`, sharing `SDG_MIRROR`'s
  strip-set with the clone path.
- **Mirror warning** (`_sdg_signature`) is a **curated** comparison that deliberately
  ignores seed / sample count (an eval set regenerates with a fresh seed yet is "the same
  recipe"). Merging it with the blanket hash would make a fresh-seed eval set falsely fail
  the mirror check. It **stays in the API layer** next to its diff messages.

So the module unifies the **clone + strip-sets** (one source of truth) and the **strict
gate signature**; the curated recipe-sameness comparison is left distinct on purpose.

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

## Core idea: one strip-set per job type

Every mirror operation is `source config → derived config` under a **field policy**. Each
job type declares one `strip_keys` set — the fields a clone never carries forward:

- **runtime** — worker-injected at dispatch, recomputed by the config builder (this is what
  `_RETRY_STRIP_KEYS` is, generalized).
- **semantic** — sample-size / mode knobs the caller is *expected* to re-set (SDG
  `{num_records, mode}`); re-introduced via `overrides`.
- **lineage** — `parent_job_id` / `training_job_id`, re-set by the create path.

Everything else — the recipe — carries forward automatically. The **clone** gains
drift-safety from this (a new recipe field is carried without code changes). The **strict
gate signature** uses the *same* `strip_keys`, so "what is the recipe" is defined once for
clone + gate. (The curated recipe-*sameness* signature for the mirror warning is a separate,
deliberate judgment — see the correction note at the top — and is not derived from this set.)

## Module `src/amortized/core/mirror.py` (pure — no FastAPI/opencode deps)

As built, `MirrorSpec` carries a single `strip_keys` set (runtime + semantic + lineage
fields a clone never carries). Keeping it to one set — rather than the runtime/semantic
split the earlier draft proposed — matched the two real call sites, which each strip a
flat set; overrides re-introduce whatever the caller changes.

```python
@dataclass(frozen=True)
class MirrorSpec:
    strip_keys: frozenset[str]          # never carried by clone (runtime + semantic + lineage)

def clone(src, spec, *, overrides=None, also_strip=()) -> dict:
    """Carry src minus strip_keys (+also_strip), drop None values, apply overrides. Deep-copied."""

def strict_signature(src, spec) -> str:
    """Blanket hash of everything carried (all − strip_keys). For the preview→full gate."""
```

Per-job-type specs are the single source of truth that replaces the three ad-hoc drop-sets
(`non_recipe_keys`, `_RETRY_STRIP_KEYS`, `_SDG_NON_RECIPE_FIELDS`):

```python
SDG_MIRROR      = MirrorSpec(strip_keys={"num_records", "mode", "parent_job_id"})
TRAINING_MIRROR = MirrorSpec(strip_keys={"parent_job_id"})
EVAL_MIRROR     = MirrorSpec(strip_keys={"eval_data_path", "port", "served_model_name", "parent_job_id"})
MIRROR_SPECS    = {"sdg": SDG_MIRROR, "training": TRAINING_MIRROR, "eval": EVAL_MIRROR}
```

The curated recipe-sameness signature + `diff_recipe` for the mirror *warning* stay in the
API layer (`_sdg_signature` / `_training_sdg_mirror_warning`) — see the correction note above.

**Hard boundary:** `core/mirror.py` is pure. The API route module (`jobs.py`) and the proxy
(`agent_confirm.py`) both *import* it; neither imports the other. The proxy keeps only its
tool-call-part→config adapter locally and reuses `strict_signature`.

## How the six use cases map onto it

| # | Use case | Covered by | Notes |
|---|---|---|---|
| 1 | **Generalized retry/rerun** (SDG, training, eval) | `clone(src, spec)` from the `request_config` snapshot | lift the eval-only restriction; always via a confirm card (see guardrails) |
| 2 | **Rerun with a tweak** ("train again, 5 epochs" / "regenerate, 2000 records") | `clone(src, spec, overrides={...})` | overrides are exactly the `semantic_keys` the caller sets; shown in the confirm card |
| 3 | **Preview → full-run** | `strict_signature` (= `_sdg_config_signature`) | shares `SDG_MIRROR` strip-set with clone; `mode`/count stripped so preview≈full. **Done.** |
| 4 | **Split siblings** (`split_run_id` / `complement_run_id`) | curated signature equality across the two runs' source configs | expresses "same recipe, different partition" as data. *Follow-up.* |
| 5 | **Cross-model eval on the same eval set** | `clone(eval_cfg, EVAL_MIRROR, overrides={"training_job_id": B})` | mirror an eval, swap only the model under test. *Follow-up (UI).* |
| 6 | **Generalized drift warning** | generalize `_training_sdg_mirror_warning`'s diff | any declared-source relationship can warn on divergence. *Follow-up.* |

Cases 1–3 are **done** (collapse the live duplication + unlock SDG/training retry/rerun).
4–6 reuse the same primitives on new call sites and are follow-ups.

## Guardrails that must NOT be folded into the module (policy ≠ mechanics)

The module unifies the *plumbing*. The *decisions* stay at the call sites:

- **Every rerun routes through a confirmation step — for all types, identical or with
  overrides.** Rerun is allowed for every type including training (confirmed); but because
  any rerun can burn real GPU, none of them dispatches on a single click.
  - **As built:** the jobs-detail-panel Retry button (the only retry surface today) now
    opens a **confirmation dialog** before calling `retry_job` — replacing the former
    one-click eval retry. The dialog's copy adapts per type (eval → "same comparison group";
    training → "consumes GPU for the full run"; SDG → "regenerates the dataset").
  - The `retry_job` endpoint itself is generalized (all types, optional `overrides`) and
    still does the create; the confirmation gate lives at the call site (the dialog), per
    *policy ≠ mechanics*. A future chat-driven rerun would instead seed a `validate_*` card.
  - An override-editing UI (case 2 in the panel) is a follow-up; the endpoint already
    accepts `overrides`.
- **`retry_of` lineage** continues to be set by the create path, not the module.
- **Stage-gate dependency invariant** (upstream job must be `succeeded`) is unchanged — a
  rerun of a downstream job still can't advance past a non-succeeded upstream.

## Call sites rewired (done)

1. `clone_sdg_config_for_eval` → `mirror.clone(parent_cfg, SDG_MIRROR, overrides={"num_records": n})`. ✅
2. `retry_job` → generalized: dropped the eval-only check; `mirror.clone(snapshot, MIRROR_SPECS[type], overrides=...)`;
   per-type validation; rejects unsupported types (e.g. `serve`). Confirmation gated by the
   jobs-panel dialog. ✅
3. `_sdg_config_signature` (proxy preview gate) → `mirror.strict_signature(..., SDG_MIRROR)` (local part adapter kept). ✅
4. `_sdg_signature` / `_training_sdg_mirror_warning` → **left as-is** (curated recipe-sameness,
   distinct by design — see correction note).

## Tests

- `tests/test_mirror.py`: `clone` drops strip-keys / None, applies overrides, deep-copies,
  parses JSON-string sources; `strict_signature` matches preview↔full, changes on any recipe
  edit (incl. constraints), specs cover all job types. ✅
- Parity held: existing `tests/test_eval_sdg_mirror.py` (33) + gate tests pass unchanged. ✅
- `tests/test_eval_jobs.py`: updated the old eval-only test → SDG retry 201, rerun-with-
  overrides, unsupported-type 422 (DB-backed; run on CI). ✅
- Follow-up behavior: cross-model eval mirror; an override-editing panel UI.

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
