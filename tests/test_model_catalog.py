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


class TestProviderInjection:
    """The native Anthropic + BYOK MaaS providers the data-designer builtin catalog lacks."""

    def test_maas_def_when_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAAS_BASE_URL", "https://maas.example.com/v1/")
        monkeypatch.setenv("MAAS_API_KEY", "sk-oai-x")
        assert model_catalog._maas_provider_def() == {
            "name": "maas",
            "endpoint": "https://maas.example.com/v1",  # trailing slash trimmed
            "provider_type": "openai",
            "api_key": "MAAS_API_KEY",
        }

    def test_maas_def_absent_without_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAAS_BASE_URL", "https://maas.example.com/v1")
        monkeypatch.delenv("MAAS_API_KEY", raising=False)
        assert model_catalog._maas_provider_def() is None

    def test_maas_def_absent_when_not_https(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The server sends the bearer key to this URL — a plaintext endpoint is refused so the
        # credential never crosses http.
        monkeypatch.setenv("MAAS_BASE_URL", "http://maas.example.com/v1")
        monkeypatch.setenv("MAAS_API_KEY", "sk-oai-x")
        assert model_catalog._maas_provider_def() is None

    def test_anthropic_def_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        d = model_catalog._anthropic_provider_def()
        assert d is not None
        assert d["name"] == "anthropic"
        assert d["provider_type"] == "anthropic"
        assert d["api_key"] == "ANTHROPIC_API_KEY"

    def test_anthropic_def_absent_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert model_catalog._anthropic_provider_def() is None

    def test_enabled_defs_inject_anthropic_and_maas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Independent of the data-designer builtins: the injected providers appear in the set.
        monkeypatch.setattr(
            model_catalog, "_anthropic_provider_def",
            lambda: {"name": "anthropic", "endpoint": "https://api.anthropic.com/v1",
                     "provider_type": "anthropic", "api_key": "ANTHROPIC_API_KEY"},
        )
        monkeypatch.setattr(
            model_catalog, "_maas_provider_def",
            lambda: {"name": "maas", "endpoint": "https://maas.example.com/v1",
                     "provider_type": "openai", "api_key": "MAAS_API_KEY"},
        )
        names = [d["name"] for d in model_catalog.enabled_provider_defs()]
        assert "anthropic" in names
        assert "maas" in names


async def test_accessible_model_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    defs = [
        {"name": "openai", "endpoint": "https://api.openai.com/v1",
         "provider_type": "openai", "api_key": "OPENAI_API_KEY"},
    ]
    monkeypatch.setattr(model_catalog, "enabled_provider_defs", lambda: defs)

    async def fake_fetch(pdef: dict) -> list[str]:
        return ["gpt-5.6-sol", "gpt-4o"]

    monkeypatch.setattr(model_catalog, "_fetch_provider_models", fake_fetch)
    assert await model_catalog.accessible_model_ids() == {"openai": {"gpt-5.6-sol", "gpt-4o"}}
