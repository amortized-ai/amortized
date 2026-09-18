"""Tests for MLflowClient gateway-model mapping."""

from __future__ import annotations

import pytest

from amortized.core.mlflow_client import MLflowClient


@pytest.mark.asyncio
async def test_list_gateway_models_reports_gateway_provider(monkeypatch) -> None:
    """Gateway-served models must report provider 'gateway' (the provider a job
    uses to reach them), not the endpoint's underlying type.

    The SDG skill copies the provider verbatim (<from-list_models>). On a
    gateway-only cluster the underlying type (e.g. 'openai') is unresolvable, so
    reporting it makes SDG fail with "No provider named 'openai' registered". The
    underlying model stays in model_name.
    """
    client = MLflowClient("http://mlflow:5000")

    async def fake_endpoints() -> list[dict]:
        return [
            {
                "name": "gpt-oss",
                "model_mappings": [
                    {
                        "model_definition": {
                            "provider": "openai",
                            "model_name": "openai/gpt-oss-120b",
                        }
                    }
                ],
            }
        ]

    monkeypatch.setattr(client, "list_gateway_endpoints", fake_endpoints)

    models = await client.list_gateway_models()

    assert models == [
        {"name": "gpt-oss", "provider": "gateway", "model_name": "openai/gpt-oss-120b"}
    ]
