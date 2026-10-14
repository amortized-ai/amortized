"""Shared config-mirror primitives.

"Derive a new job config from an existing one — carry the parts that must stay
identical, change only what's meant to change" is the shape of three operations:
clone-for-eval, retry/rerun, and the preview→full-run confirm gate. This module is
the one place that knows, per job type, which fields are *not* carried forward
(runtime fields the worker recomputes + sample-size / lineage fields the caller
re-sets). Pure: no FastAPI / opencode / DB deps, so both the API routes and the
proxy can import it (neither imports the other).

NOTE on signatures: there are intentionally TWO, with different strictness, and
they must NOT be merged:
  * `strict_signature` — a blanket hash of the whole carried recipe. Used by the
    preview→full gate: ANY recipe edit since the approved preview must re-gate, so
    it folds in everything (constraints, tool_configs, seed, …).
  * recipe *sameness* for the mirror warning stays a curated, structured comparison
    (teacher / prompts / topic / documents / columns) that deliberately IGNORES
    seed and sample count — an eval set regenerates with a fresh seed yet is still
    "the same recipe." That lives next to its diff messages in the API layer.
Collapsing the two would make a fresh-seed eval set falsely fail the mirror check
(or weaken the preview gate), so they are kept distinct on purpose.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


def coerce_config(raw: Any) -> dict[str, Any]:
    """A config may arrive as a dict or a JSON string (DB column / tool input).
    Return a dict, or an empty dict when unparseable."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


@dataclass(frozen=True)
class MirrorSpec:
    """How to mirror one job type's config forward.

    `strip_keys` are the keys `clone` never carries: worker-injected runtime fields
    (recomputed by the config builder) plus the sample-size / lineage knobs the
    caller re-sets (num_records, mode, parent_job_id, …). This is the single source
    of truth that replaces the per-call-site ad-hoc drop-sets.
    """

    strip_keys: frozenset[str]


def clone(
    src: Any,
    spec: MirrorSpec,
    *,
    overrides: Mapping[str, Any] | None = None,
    also_strip: Iterable[str] = (),
) -> dict[str, Any]:
    """Carry `src` forward minus `spec.strip_keys` (plus any `also_strip`), dropping
    None-valued keys, then apply `overrides` on top. Deep-copied, so the result
    shares no mutable state with `src`.

    `overrides` are exactly the semantic knobs the caller intends to change
    ("regenerate, 2000 records" → overrides={"num_records": 2000}); an identical
    rerun passes none.
    """
    cfg = coerce_config(src)
    drop = spec.strip_keys | frozenset(also_strip)
    out = {
        key: copy.deepcopy(value)
        for key, value in cfg.items()
        if key not in drop and value is not None
    }
    for key, value in (overrides or {}).items():
        out[key] = copy.deepcopy(value)
    return out


def strict_signature(src: Any, spec: MirrorSpec) -> str:
    """Blanket hash of everything carried forward (all keys minus `strip_keys`).

    Used by the preview→full-run gate: identical for a preview and the full-job
    confirm of the same recipe (they differ only in the stripped mode / record
    count), but ANY other recipe edit yields a new signature so the gate re-fires.
    """
    recipe = {k: v for k, v in coerce_config(src).items() if k not in spec.strip_keys}
    canonical = json.dumps(recipe, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Per-job-type specs — the one place each type's non-carried fields are declared.
#
# SDG: a preview approves the RECIPE; num_records/mode count a sample, parent_job_id
# is lineage. (Mirrors the clone-for-eval invariant "only num_records changes".)
SDG_MIRROR = MirrorSpec(strip_keys=frozenset({"num_records", "mode", "parent_job_id"}))

# Training: lineage is re-set by the create path; the recipe (algo/hyperparams/base
# model) carries. Overrides handle "train again, 5 epochs".
TRAINING_MIRROR = MirrorSpec(strip_keys=frozenset({"parent_job_id"}))

# Eval: eval_data_path/port/served_model_name are worker-injected at dispatch;
# parent_job_id / training_job_id are lineage re-set by the create path.
EVAL_MIRROR = MirrorSpec(
    strip_keys=frozenset({"eval_data_path", "port", "served_model_name", "parent_job_id"})
)

# Lookup by job-type string (JobType.value) for the generalized retry/rerun path.
MIRROR_SPECS: dict[str, MirrorSpec] = {
    "sdg": SDG_MIRROR,
    "training": TRAINING_MIRROR,
    "eval": EVAL_MIRROR,
}
