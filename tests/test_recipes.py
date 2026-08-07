"""Tests for the recipe system — core logic and API endpoints."""

import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from amortized.core.recipes import (
    CircularRecipeError,
    RecipeNotFoundError,
    apply_overrides,
    list_recipes,
    load_recipe,
)


@pytest.fixture
def recipes_dir(tmp_path: Path) -> Path:
    training = tmp_path / "training"
    training.mkdir()
    (training / "lora-sft.yaml").write_text(
        "type: training\n"
        "description: LoRA SFT\n"
        "config:\n"
        "  algorithm: sft\n"
        "  num_train_epochs: 3\n"
        "  learning_rate: 0.0002\n"
        "  lora_r: 16\n"
        "  lora_alpha: 32\n"
    )
    (training / "grpo.yaml").write_text(
        "type: training\ndescription: GRPO RL\nconfig:\n  algorithm: grpo\n  num_iterations: 15\n"
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "llama3-8b-lora.yaml").write_text(
        "extends: training/lora-sft\n"
        "description: Llama 3.1 8B LoRA\n"
        "config:\n"
        "  model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
        "  max_length: 8192\n"
    )
    return tmp_path


class TestLoadRecipe:
    def test_load_base_recipe(self, recipes_dir: Path) -> None:
        recipe = load_recipe("training/lora-sft", recipes_dir=recipes_dir)
        assert recipe["type"] == "training"
        assert recipe["description"] == "LoRA SFT"
        assert recipe["config"]["num_train_epochs"] == 3
        assert recipe["config"]["lora_r"] == 16

    def test_load_nonexistent_raises(self, recipes_dir: Path) -> None:
        with pytest.raises(RecipeNotFoundError):
            load_recipe("nonexistent/recipe", recipes_dir=recipes_dir)

    def test_extends_merges_parent(self, recipes_dir: Path) -> None:
        recipe = load_recipe("models/llama3-8b-lora", recipes_dir=recipes_dir)
        assert recipe["type"] == "training"
        assert recipe["description"] == "Llama 3.1 8B LoRA"
        assert recipe["config"]["model_name_or_path"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert recipe["config"]["max_length"] == 8192
        assert recipe["config"]["num_train_epochs"] == 3
        assert recipe["config"]["lora_r"] == 16
        assert "extends" not in recipe

    def test_circular_extends_raises(self, recipes_dir: Path) -> None:
        (recipes_dir / "a.yaml").write_text("extends: b\ntype: training\ndescription: A\n")
        (recipes_dir / "b.yaml").write_text("extends: a\ntype: training\ndescription: B\n")
        with pytest.raises(CircularRecipeError):
            load_recipe("a", recipes_dir=recipes_dir)

    def test_self_extending_raises(self, recipes_dir: Path) -> None:
        (recipes_dir / "self.yaml").write_text("extends: self\ntype: training\ndescription: Self\n")
        with pytest.raises(CircularRecipeError):
            load_recipe("self", recipes_dir=recipes_dir)

    def test_configurable_recipes_dir(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "custom_recipes"
        custom_dir.mkdir()
        (custom_dir / "custom.yaml").write_text(
            "type: training\ndescription: Custom recipe\nconfig:\n  num_train_epochs: 5\n"
        )
        import amortized.config as config_mod

        old_dir = config_mod.settings.recipes_dir
        config_mod.settings.recipes_dir = custom_dir
        try:
            recipe = load_recipe("custom")
            assert recipe["description"] == "Custom recipe"
            assert recipe["config"]["num_train_epochs"] == 5
        finally:
            config_mod.settings.recipes_dir = old_dir


class TestListRecipes:
    def test_list_all(self, recipes_dir: Path) -> None:
        recipes = list_recipes(recipes_dir=recipes_dir)
        names = {r["name"] for r in recipes}
        assert "training/lora-sft" in names
        assert "training/grpo" in names
        assert "models/llama3-8b-lora" in names

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert list_recipes(recipes_dir=empty) == []

    def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        assert list_recipes(recipes_dir=tmp_path / "nope") == []


class TestApplyOverrides:
    def test_dotted_key_override(self) -> None:
        recipe: dict[str, Any] = {"type": "training", "config": {"num_train_epochs": 3}}
        result = apply_overrides(recipe, {"config.num_train_epochs": 5})
        assert result["config"]["num_train_epochs"] == 5

    def test_creates_nested_keys(self) -> None:
        recipe: dict[str, Any] = {"type": "training"}
        result = apply_overrides(recipe, {"config.data_path": "/data/train.jsonl"})
        assert result["config"]["data_path"] == "/data/train.jsonl"

    def test_preserves_unaffected_keys(self) -> None:
        recipe: dict[str, Any] = {"type": "training", "config": {"a": 1, "b": 2}}
        result = apply_overrides(recipe, {"config.a": 10})
        assert result["config"]["a"] == 10
        assert result["config"]["b"] == 2


TEST_DATABASE_URL = os.environ.get(
    "AMORTIZED_TEST_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized_test",
)


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: Path) -> None:
    import amortized.config as config_mod
    import amortized.db.connection as db_conn_mod

    os.environ["AMORTIZED_DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_conn_mod.settings = new_settings


class TestRecipeAPI:
    @pytest.fixture
    async def client(self) -> httpx.AsyncClient:  # type: ignore[misc]
        from amortized.db import init_db
        from amortized.main import app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            await init_db()
            yield c  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_list_recipes_endpoint(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/recipes")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = {r["name"] for r in data}
        assert "templates/training/knowledge-ingestion" in names
        assert "templates/sdg/knowledge-ingestion" in names

    @pytest.mark.asyncio
    async def test_get_recipe_endpoint(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/recipes/templates/training/knowledge-ingestion")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "training"
        assert "config" in data

    @pytest.mark.asyncio
    async def test_get_nonexistent_recipe(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/recipes/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_submit_recipe_job(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/jobs/recipe",
            json={
                "recipe": "templates/training/knowledge-ingestion",
                "overrides": {
                    "config.model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "config.data_path": "/data/train.jsonl",
                },
                "dry_run": False,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["type"] == "training"
        assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_submit_recipe_validates_config(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/jobs/recipe",
            json={
                "recipe": "templates/training/knowledge-ingestion",
                "overrides": {
                    "config.model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "config.data_path": "/data/train.jsonl",
                },
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["type"] == "training"

    @pytest.mark.asyncio
    async def test_submit_sdg_flat_overrides_reach_config(self, client: httpx.AsyncClient) -> None:
        """Flat overrides sent by the UI must land in the stored job config."""
        resp = await client.post(
            "/api/v1/jobs/recipe",
            json={
                "recipe": "templates/sdg/knowledge-ingestion",
                "overrides": {
                    "config.num_records": 20,
                },
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["type"] == "sdg"
        assert data["config"]["num_records"] == 20

    @pytest.mark.asyncio
    async def test_submit_sdg_empty_overrides_dont_clobber(self, client: httpx.AsyncClient) -> None:
        """Empty/falsy overrides from the form should not replace the
        recipe's own defaults."""
        resp = await client.post(
            "/api/v1/jobs/recipe",
            json={
                "recipe": "templates/sdg/knowledge-ingestion",
                "overrides": {
                    "config.num_records": 50,
                },
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "columns" in data["config"]
        assert data["config"]["num_records"] == 50

    @pytest.mark.asyncio
    async def test_submit_nonexistent_recipe(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/jobs/recipe",
            json={"recipe": "nonexistent/recipe"},
        )
        assert resp.status_code == 404
