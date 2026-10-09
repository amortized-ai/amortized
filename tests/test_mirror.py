"""Unit tests for the shared config-mirror primitives (core/mirror.py)."""

import json

from amortized.core import mirror


def _sdg_cfg() -> dict:
    return {
        "num_records": 100,
        "mode": "create",
        "parent_job_id": "sdg-123",
        "model_configs": [{"alias": "teacher", "model": "gpt-oss"}],
        "columns": [{"column_type": "llm-text", "name": "out", "system_prompt": "assess"}],
        "constraints": {"max_tokens": 512},
        "tool_configs": [{"name": "calc"}],
    }


class TestClone:
    def test_drops_strip_keys_and_carries_recipe(self) -> None:
        out = mirror.clone(_sdg_cfg(), mirror.SDG_MIRROR)
        assert "num_records" not in out  # stripped (semantic)
        assert "mode" not in out
        assert "parent_job_id" not in out
        # recipe-defining fields carry over — including ones a denylist must not drop
        assert out["constraints"] == {"max_tokens": 512}
        assert out["tool_configs"] == [{"name": "calc"}]
        assert out["model_configs"][0]["model"] == "gpt-oss"

    def test_overrides_applied_on_top(self) -> None:
        out = mirror.clone(_sdg_cfg(), mirror.SDG_MIRROR, overrides={"num_records": 50})
        assert out["num_records"] == 50  # stripped then re-set by the override

    def test_override_can_change_a_carried_field(self) -> None:
        # "train again, 5 epochs" — the rerun-with-a-tweak case.
        cfg = {"num_train_epochs": 3, "algo": "sft", "parent_job_id": "t-1"}
        out = mirror.clone(cfg, mirror.TRAINING_MIRROR, overrides={"num_train_epochs": 5})
        assert out["num_train_epochs"] == 5
        assert out["algo"] == "sft"
        assert "parent_job_id" not in out

    def test_drops_none_values(self) -> None:
        out = mirror.clone({"a": 1, "b": None}, mirror.TRAINING_MIRROR)
        assert out == {"a": 1}

    def test_also_strip(self) -> None:
        out = mirror.clone({"a": 1, "b": 2}, mirror.TRAINING_MIRROR, also_strip=["b"])
        assert out == {"a": 1}

    def test_deep_copies(self) -> None:
        src = _sdg_cfg()
        out = mirror.clone(src, mirror.SDG_MIRROR)
        out["constraints"]["max_tokens"] = 9999
        assert src["constraints"]["max_tokens"] == 512  # source untouched

    def test_accepts_json_string(self) -> None:
        out = mirror.clone(json.dumps({"a": 1, "parent_job_id": "x"}), mirror.SDG_MIRROR)
        assert out == {"a": 1}

    def test_unparseable_source_is_empty(self) -> None:
        assert mirror.clone("not json", mirror.SDG_MIRROR) == {}
        assert mirror.clone(None, mirror.SDG_MIRROR) == {}


class TestStrictSignature:
    def test_preview_and_full_run_match(self) -> None:
        # Preview vs full confirm of the SAME recipe differ only in mode/record count,
        # which are stripped — so the gate sees them as the same recipe.
        preview = {**_sdg_cfg(), "mode": "preview", "num_records": 10}
        full = {**_sdg_cfg(), "mode": "create", "num_records": 5000}
        assert mirror.strict_signature(preview, mirror.SDG_MIRROR) == mirror.strict_signature(
            full, mirror.SDG_MIRROR
        )

    def test_recipe_edit_changes_signature(self) -> None:
        base = _sdg_cfg()
        edited = {**base, "model_configs": [{"alias": "teacher", "model": "gpt-4o"}]}
        assert mirror.strict_signature(base, mirror.SDG_MIRROR) != mirror.strict_signature(
            edited, mirror.SDG_MIRROR
        )

    def test_strict_catches_constraint_edit(self) -> None:
        # Unlike the curated mirror-warning signature, the gate must re-fire on ANY
        # recipe edit — including constraints/tool_configs a denylist clone carries.
        base = _sdg_cfg()
        edited = {**base, "constraints": {"max_tokens": 1024}}
        assert mirror.strict_signature(base, mirror.SDG_MIRROR) != mirror.strict_signature(
            edited, mirror.SDG_MIRROR
        )


class TestSpecs:
    def test_mirror_specs_cover_all_job_types(self) -> None:
        assert set(mirror.MIRROR_SPECS) == {"sdg", "training", "eval"}

    def test_eval_strips_runtime_and_lineage(self) -> None:
        assert {"eval_data_path", "port", "served_model_name", "parent_job_id"} <= (
            mirror.EVAL_MIRROR.strip_keys
        )
