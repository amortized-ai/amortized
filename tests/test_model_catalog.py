"""Tests for the direct-provider model catalog: chat filtering + the dynamic pull."""

import pytest

from amortized.core import model_catalog


class TestKeepModel:
    def test_openai_keeps_gpt5_gpt6_families(self) -> None:
        for mid in ["gpt-5", "gpt-5.6-sol", "gpt-5-mini", "gpt-6", "gpt-6-astra", "gpt-6-sol"]:
            assert model_catalog._keep_model("openai", mid), mid

    def test_openai_drops_older_and_non_chat(self) -> None:
        for mid in [
            "gpt-4o",
            "gpt-4.1",
            "o3-mini",
            "text-embedding-3-large",
            "dall-e-3",
            "whisper-1",
        ]:
            assert not model_catalog._keep_model("openai", mid), mid

    def test_other_providers_keep_chat_drop_non_chat(self) -> None:
        # Non-openai providers: any chat model kept (no prefix rule), non-chat dropped by marker.
        assert model_catalog._keep_model("nvidia", "nvidia/nemotron-3-super-120b-a12b")
        assert model_catalog._keep_model("openrouter", "anthropic/claude-3.5-sonnet")
        assert not model_catalog._keep_model("nvidia", "nvidia/llama-nemotron-embed-1b-v2")
        assert not model_catalog._keep_model("openrouter", "some/rerank-model")

    def test_empty_id_dropped(self) -> None:
        assert not model_catalog._keep_model("openai", "")


async def test_available_models_pulls_filters_and_dedups(monkeypatch: pytest.MonkeyPatch) -> None:
    defs = [
        {
            "name": "openai",
            "endpoint": "https://api.openai.com/v1",
            "provider_type": "openai",
            "api_key": "OPENAI_API_KEY",
        },
        {
            "name": "nvidia",
            "endpoint": "https://integrate.api.nvidia.com/v1",
            "provider_type": "openai",
            "api_key": "NVIDIA_API_KEY",
        },
    ]
    raw = {
        "openai": ["gpt-5.6-sol", "gpt-6-astra", "gpt-4o", "text-embedding-3-large", "gpt-5.6-sol"],
        "nvidia": ["nvidia/nemotron-3-super-120b-a12b", "nvidia/llama-nemotron-embed-1b-v2"],
    }
    monkeypatch.setattr(model_catalog, "enabled_provider_defs", lambda: defs)

    async def fake_fetch(pdef: dict) -> list[str]:
        return raw[pdef["name"]]

    monkeypatch.setattr(model_catalog, "_fetch_provider_models", fake_fetch)

    out = await model_catalog.available_models()
    # openai filtered to gpt-5/gpt-6 (gpt-4o + embedding dropped) and deduped; nvidia chat only.
    assert out == [
        ("openai", "gpt-5.6-sol"),
        ("openai", "gpt-6-astra"),
        ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
    ]


async def test_available_models_empty_when_no_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_catalog, "enabled_provider_defs", lambda: [])
    assert await model_catalog.available_models() == []
