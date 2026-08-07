"""Adversarial tests for POST /api/v1/jobs/sdg — the typed SDG endpoint.

Covers every validation error case from _simplify_sdg_errors and the
SDGJobRequest model validator, ensuring Morty gets clear, actionable
error messages it can self-correct from in one retry.
"""

import os

import httpx
import pytest

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db.connection as db_conn_mod

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


VALID_LLM_TEXT_COLUMN = {
    "column_type": "llm-text",
    "name": "question",
    "model_alias": "text",
    "prompt": "Generate a question about {{ content }}",
}

VALID_MODEL_CONFIG = {"alias": "text", "model": "gpt-4o", "provider": "openrouter"}

VALID_SAMPLER_COLUMN = {
    "column_type": "sampler",
    "name": "difficulty",
    "sampler_type": "category",
    "params": {"values": ["Easy", "Medium", "Hard"]},
}


def _get_detail(response: httpx.Response) -> list[dict[str, str]]:
    """Extract detail list, handling both response formats.

    The error handler may wrap HTTPException detail into:
      {"code": "http_422", "message": "<stringified list>", "details": []}
    or the raw FastAPI format:
      {"detail": [{"field": ..., "error": ...}]}
    """
    body = response.json()
    if "detail" in body and isinstance(body["detail"], list):
        return body["detail"]
    if "message" in body and isinstance(body["message"], str):
        import ast

        try:
            parsed = ast.literal_eval(body["message"])
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def _err_fields(response: httpx.Response) -> list[str]:
    """Extract field names from error response."""
    return [e["field"] for e in _get_detail(response) if "field" in e]


def _err_messages(response: httpx.Response) -> list[str]:
    """Extract error messages from error response."""
    return [e["error"] for e in _get_detail(response) if "error" in e]


def _err_str(response: httpx.Response) -> str:
    """Full error response as string for substring matching."""
    return str(response.json())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestSDGEndpointHappyPath:
    @pytest.mark.asyncio
    async def test_minimal_sampler_only(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_SAMPLER_COLUMN],
                "num_records": 10,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "sdg"
        assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_llm_text_with_model_config(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_LLM_TEXT_COLUMN],
                "model_configs": [VALID_MODEL_CONFIG],
                "num_records": 5,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "sdg"
        assert data["config"]["num_records"] == 5

    @pytest.mark.asyncio
    async def test_multi_column_pipeline(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    VALID_SAMPLER_COLUMN,
                    {
                        **VALID_LLM_TEXT_COLUMN,
                        "prompt": "Difficulty: {{ difficulty }}. Generate a question.",
                    },
                ],
                "model_configs": [VALID_MODEL_CONFIG],
                "num_records": 20,
            },
        )
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_preview_mode(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_SAMPLER_COLUMN],
                "mode": "preview",
            },
        )
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_with_topic_and_parent(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_SAMPLER_COLUMN],
                "topic": "OpenShift QA",
                "parent_job_id": "parent-123",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["parent_job_id"] == "parent-123"


# ---------------------------------------------------------------------------
# Missing / empty required fields
# ---------------------------------------------------------------------------


class TestSDGMissingFields:
    @pytest.mark.asyncio
    async def test_missing_columns(self, client: httpx.AsyncClient) -> None:
        """Empty body → columns: Field required."""
        response = await client.post("/api/v1/jobs/sdg", json={})
        assert response.status_code == 422
        assert "columns" in _err_str(response)
        assert "required" in _err_str(response).lower()

    @pytest.mark.asyncio
    async def test_empty_columns_list(self, client: httpx.AsyncClient) -> None:
        """columns: [] → accepted (DD allows empty columns list)."""
        response = await client.post("/api/v1/jobs/sdg", json={"columns": []})
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_missing_prompt_on_llm_text(self, client: httpx.AsyncClient) -> None:
        """llm-text column without prompt → columns[0].prompt: Field required."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                    }
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        assert "prompt" in _err_str(response)
        assert "required" in _err_str(response).lower()

    @pytest.mark.asyncio
    async def test_missing_model_alias_on_llm_text(self, client: httpx.AsyncClient) -> None:
        """llm-text without model_alias → columns[0].model_alias: Field required."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "prompt": "Generate a question.",
                    }
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        assert "model_alias" in _err_str(response)

    @pytest.mark.asyncio
    async def test_missing_sampler_params(self, client: httpx.AsyncClient) -> None:
        """Sampler without params → columns[0].params: Field required."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "sampler",
                        "name": "difficulty",
                        "sampler_type": "category",
                    }
                ],
            },
        )
        assert response.status_code == 422
        assert "params" in _err_str(response)

    @pytest.mark.asyncio
    async def test_missing_name_on_column(self, client: httpx.AsyncClient) -> None:
        """Column without name → columns[0].name: Field required."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "sampler",
                        "sampler_type": "uuid",
                        "params": {},
                    }
                ],
            },
        )
        assert response.status_code == 422
        assert "name" in _err_str(response)


# ---------------------------------------------------------------------------
# Invalid column_type
# ---------------------------------------------------------------------------


class TestSDGInvalidColumnType:
    @pytest.mark.asyncio
    async def test_invalid_column_type(self, client: httpx.AsyncClient) -> None:
        """'llm-chat' → not valid, shows valid types list."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [{"column_type": "llm-chat", "name": "q", "prompt": "hi"}],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "llm-chat" in errors
        assert "not valid" in errors.lower() or "is not valid" in errors.lower()
        assert "llm-text" in errors

    @pytest.mark.asyncio
    async def test_typo_column_type_underscore(self, client: httpx.AsyncClient) -> None:
        """'llm_text' (underscore) → not valid."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [{"column_type": "llm_text", "name": "q", "prompt": "hi"}],
            },
        )
        assert response.status_code == 422
        assert "llm_text" in _err_str(response)
        assert "not valid" in _err_str(response).lower()

    @pytest.mark.asyncio
    async def test_completely_bogus_column_type(self, client: httpx.AsyncClient) -> None:
        """'magic-generator' → not valid."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [{"column_type": "magic-generator", "name": "q"}],
            },
        )
        assert response.status_code == 422
        assert "magic-generator" in _err_str(response)

    @pytest.mark.asyncio
    async def test_multiple_bad_column_types(self, client: httpx.AsyncClient) -> None:
        """Two columns with bad types → error for each."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {"column_type": "llm-chat", "name": "a"},
                    {"column_type": "foo-bar", "name": "b"},
                ],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "columns[0]" in errors
        assert "columns[1]" in errors
        assert "llm-chat" in errors
        assert "foo-bar" in errors


# ---------------------------------------------------------------------------
# Model alias cross-referencing
# ---------------------------------------------------------------------------


class TestSDGModelAlias:
    @pytest.mark.asyncio
    async def test_model_alias_mismatch(self, client: httpx.AsyncClient) -> None:
        """model_alias 'teacher' not in model_configs → clear error."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "teacher",
                        "prompt": "Generate a question.",
                    }
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "teacher" in errors
        assert "text" in errors

    @pytest.mark.asyncio
    async def test_model_configs_missing_when_llm_column_used(
        self, client: httpx.AsyncClient
    ) -> None:
        """llm-text column without any model_configs → tells you it's required."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                        "prompt": "Generate a question.",
                    }
                ],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "model_configs" in errors
        assert "required" in errors.lower()

    @pytest.mark.asyncio
    async def test_multiple_aliases_one_missing(self, client: httpx.AsyncClient) -> None:
        """Two llm columns, one alias present and one missing."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                        "prompt": "Generate a question.",
                    },
                    {
                        "column_type": "llm-text",
                        "name": "answer",
                        "model_alias": "judge",
                        "prompt": "Answer: {{ question }}",
                    },
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "judge" in errors


# ---------------------------------------------------------------------------
# Invalid sampler_type
# ---------------------------------------------------------------------------


class TestSDGInvalidSampler:
    @pytest.mark.asyncio
    async def test_invalid_sampler_type(self, client: httpx.AsyncClient) -> None:
        """'random' is not a valid sampler_type."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "sampler",
                        "name": "val",
                        "sampler_type": "random",
                        "params": {"values": ["a"]},
                    }
                ],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "random" in errors.lower() or "bernoulli" in errors.lower()

    @pytest.mark.asyncio
    async def test_sampler_wrong_params_for_type(self, client: httpx.AsyncClient) -> None:
        """category sampler with gaussian params → error about 'values'."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "sampler",
                        "name": "val",
                        "sampler_type": "category",
                        "params": {"mean": 0.5, "stddev": 0.1},
                    }
                ],
            },
        )
        assert response.status_code == 422
        assert "values" in _err_str(response).lower()


# ---------------------------------------------------------------------------
# Extra / unknown fields (extra="forbid")
# ---------------------------------------------------------------------------


class TestSDGExtraFields:
    @pytest.mark.asyncio
    async def test_unknown_top_level_field(self, client: httpx.AsyncClient) -> None:
        """strategy_params → Extra inputs are not permitted."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_SAMPLER_COLUMN],
                "strategy_params": {"batch_size": 10},
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "strategy_params" in errors
        assert "not permitted" in errors.lower() or "extra" in errors.lower()


# ---------------------------------------------------------------------------
# Temperature / inference parameter validation
# ---------------------------------------------------------------------------


class TestSDGTemperature:
    @pytest.mark.asyncio
    async def test_temperature_too_high(self, client: httpx.AsyncClient) -> None:
        """temperature: 3.0 → must be between 0.0 and 2.0."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_LLM_TEXT_COLUMN],
                "model_configs": [
                    {
                        **VALID_MODEL_CONFIG,
                        "inference_parameters": {
                            "temperature": 3.0,
                        },
                    }
                ],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "temperature" in errors.lower() or "2.0" in errors

    @pytest.mark.asyncio
    async def test_temperature_negative(self, client: httpx.AsyncClient) -> None:
        """temperature: -0.5 → must be >= 0.0."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_LLM_TEXT_COLUMN],
                "model_configs": [
                    {
                        **VALID_MODEL_CONFIG,
                        "inference_parameters": {
                            "temperature": -0.5,
                        },
                    }
                ],
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_temperature_at_boundary(self, client: httpx.AsyncClient) -> None:
        """temperature: 2.0 → should be accepted (edge of valid range)."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_LLM_TEXT_COLUMN],
                "model_configs": [
                    {
                        **VALID_MODEL_CONFIG,
                        "inference_parameters": {
                            "temperature": 2.0,
                        },
                    }
                ],
            },
        )
        assert response.status_code == 201


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------


class TestSDGInvalidMode:
    @pytest.mark.asyncio
    async def test_invalid_mode(self, client: httpx.AsyncClient) -> None:
        """mode: 'generate' → Input should be 'create' or 'preview'."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_SAMPLER_COLUMN],
                "mode": "generate",
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "create" in errors.lower()
        assert "preview" in errors.lower()


# ---------------------------------------------------------------------------
# num_records validation
# ---------------------------------------------------------------------------


class TestSDGNumRecords:
    @pytest.mark.asyncio
    async def test_num_records_zero(self, client: httpx.AsyncClient) -> None:
        """num_records: 0 → must be >= 1."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_SAMPLER_COLUMN],
                "num_records": 0,
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_num_records_negative(self, client: httpx.AsyncClient) -> None:
        """num_records: -5 → must be >= 1."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_SAMPLER_COLUMN],
                "num_records": -5,
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_num_records_string(self, client: httpx.AsyncClient) -> None:
        """num_records: 'fifty' → type error."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_SAMPLER_COLUMN],
                "num_records": "fifty",
            },
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Error message quality — Morty self-correction checks
#
# The key requirement: each error is 1-2 lines with the exact field
# and what's wrong, so Morty can self-correct in one retry.
# ---------------------------------------------------------------------------


class TestSDGErrorMessageQuality:
    @pytest.mark.asyncio
    async def test_error_has_field_and_error_keys(self, client: httpx.AsyncClient) -> None:
        """Every error dict has 'field' and 'error' keys."""
        response = await client.post("/api/v1/jobs/sdg", json={})
        assert response.status_code == 422
        detail = _get_detail(response)
        assert len(detail) > 0, f"Expected errors, got: {response.json()}"
        for err in detail:
            assert "field" in err, f"Error missing 'field' key: {err}"
            assert "error" in err, f"Error missing 'error' key: {err}"

    @pytest.mark.asyncio
    async def test_field_path_includes_index(self, client: httpx.AsyncClient) -> None:
        """Column errors include the column index in the field path."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                    }
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        fields = _err_fields(response)
        assert any("columns[0]" in f for f in fields)

    @pytest.mark.asyncio
    async def test_no_union_explosion(self, client: httpx.AsyncClient) -> None:
        """A single missing field should NOT produce 10+ errors from all union variants."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                    }
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        detail = _get_detail(response)
        assert len(detail) <= 5, (
            f"Expected <=5 errors for a single missing field, got {len(detail)}: {detail}"
        )

    @pytest.mark.asyncio
    async def test_invalid_column_type_lists_all_valid_types(
        self, client: httpx.AsyncClient
    ) -> None:
        """Invalid column_type error lists all valid types for discovery."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [{"column_type": "wrong", "name": "q"}],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        for expected in ["llm-text", "sampler", "expression", "embedding"]:
            assert expected in errors, f"Valid type '{expected}' not listed in error"

    @pytest.mark.asyncio
    async def test_alias_mismatch_lists_available(self, client: httpx.AsyncClient) -> None:
        """Alias mismatch lists what aliases ARE available."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "q",
                        "model_alias": "wrong_alias",
                        "prompt": "hi",
                    }
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "wrong_alias" in errors
        assert "text" in errors


# ---------------------------------------------------------------------------
# Compound / multi-error scenarios
# ---------------------------------------------------------------------------


class TestSDGCompoundErrors:
    @pytest.mark.asyncio
    async def test_multiple_columns_different_errors(self, client: httpx.AsyncClient) -> None:
        """Two columns each with a different error."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "sampler",
                        "name": "diff",
                        "sampler_type": "category",
                    },
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                    },
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        errors = _err_str(response)
        assert "params" in errors.lower() or "columns[0]" in errors
        assert "prompt" in errors.lower() or "columns[1]" in errors

    @pytest.mark.asyncio
    async def test_good_and_bad_columns_mixed(self, client: httpx.AsyncClient) -> None:
        """One valid column + one invalid → only invalid column's errors."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    VALID_SAMPLER_COLUMN,
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                    },
                ],
                "model_configs": [VALID_MODEL_CONFIG],
            },
        )
        assert response.status_code == 422
        fields = _err_fields(response)
        assert not any("columns[0]" in f for f in fields), "Valid column should not have errors"
        assert any("columns[1]" in f for f in fields)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSDGEdgeCases:
    @pytest.mark.asyncio
    async def test_columns_not_a_list(self, client: httpx.AsyncClient) -> None:
        """columns: 'not a list' → type error."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={"columns": "not a list"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_column_missing_column_type(self, client: httpx.AsyncClient) -> None:
        """Column without column_type → discriminator error."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [{"name": "question", "prompt": "hi"}],
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_model_configs_not_a_list(self, client: httpx.AsyncClient) -> None:
        """model_configs: {} → type error."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [VALID_LLM_TEXT_COLUMN],
                "model_configs": {"alias": "text", "model": "gpt-4o"},
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_uuid_sampler_minimal(self, client: httpx.AsyncClient) -> None:
        """UUID sampler with empty params → valid."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "sampler",
                        "name": "id",
                        "sampler_type": "uuid",
                        "params": {},
                    }
                ],
            },
        )
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_expression_column(self, client: httpx.AsyncClient) -> None:
        """Expression column → valid, no model_configs needed."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    VALID_SAMPLER_COLUMN,
                    {
                        "column_type": "expression",
                        "name": "label",
                        "expr": "'prefix_' + difficulty",
                    },
                ],
            },
        )
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_null_body(self, client: httpx.AsyncClient) -> None:
        """null JSON body → 422 with clear message."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            content="null",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert "JSON object" in _err_str(response)

    @pytest.mark.asyncio
    async def test_empty_string_body(self, client: httpx.AsyncClient) -> None:
        """Empty string body → 422."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            content="",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_array_body(self, client: httpx.AsyncClient) -> None:
        """Array body → 422 with clear message."""
        response = await client.post(
            "/api/v1/jobs/sdg",
            content="[1, 2, 3]",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert "JSON object" in _err_str(response)
