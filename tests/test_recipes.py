"""Tests for the starter template system."""

import json
from pathlib import Path
from typing import Any

import pytest

from amortized.core.recipes import list_starter_templates


@pytest.fixture
def skills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import amortized.config as config_mod

    monkeypatch.setattr(config_mod.settings, "recipes_dir", tmp_path)

    sdg_ki = tmp_path / "agents" / "sdg" / "skills" / "knowledge-ingestion"
    sdg_ki.mkdir(parents=True)
    (sdg_ki / "reference-payload.json").write_text(
        json.dumps(
            {
                "_meta": {"name": "Knowledge Ingestion", "description": "Generate QA data"},
                "num_records": 500,
                "columns": [],
            }
        )
    )

    training_ki = tmp_path / "agents" / "training" / "skills" / "knowledge-ingestion" / "osft"
    training_ki.mkdir(parents=True)
    (training_ki / "reference-payload.json").write_text(
        json.dumps(
            {
                "_meta": {"name": "KI OSFT", "description": "Fine-tune with OSFT"},
                "algorithm": "osft",
                "model_name_or_path": "Qwen/Qwen3.5-4B",
            }
        )
    )

    return tmp_path


class TestListStarterTemplates:
    def test_returns_templates_from_skills_dir(self, skills_dir: Path) -> None:
        templates = list_starter_templates()
        assert len(templates) == 2
        names = {t["name"] for t in templates}
        assert "Knowledge Ingestion" in names
        assert "KI OSFT" in names

    def test_meta_is_stripped_from_config(self, skills_dir: Path) -> None:
        templates = list_starter_templates()
        for t in templates:
            assert "_meta" not in t["config"]

    def test_includes_type_and_use_case(self, skills_dir: Path) -> None:
        templates = list_starter_templates()
        by_name: dict[str, Any] = {t["name"]: t for t in templates}

        ki = by_name["Knowledge Ingestion"]
        assert ki["type"] == "sdg"
        assert ki["use_case"] == "knowledge-ingestion"
        assert ki["description"] == "Generate QA data"

        osft = by_name["KI OSFT"]
        assert osft["type"] == "training"
        assert osft["use_case"] == "knowledge-ingestion-osft"

    def test_empty_when_no_skills_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import amortized.config as config_mod

        monkeypatch.setattr(config_mod.settings, "recipes_dir", tmp_path)
        assert list_starter_templates() == []

    def test_skips_invalid_json(self, skills_dir: Path) -> None:
        bad = skills_dir / "agents" / "sdg" / "skills" / "broken"
        bad.mkdir(parents=True)
        (bad / "reference-payload.json").write_text("not json")
        templates = list_starter_templates()
        assert len(templates) == 2

    def test_falls_back_to_use_case_when_no_meta(self, skills_dir: Path) -> None:
        no_meta = skills_dir / "agents" / "sdg" / "skills" / "classification"
        no_meta.mkdir(parents=True)
        (no_meta / "reference-payload.json").write_text(json.dumps({"num_records": 100}))
        templates = list_starter_templates()
        names = {t["name"] for t in templates}
        assert "classification" in names
