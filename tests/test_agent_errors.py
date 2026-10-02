"""Tests for provider-error mapping surfaced in Morty chat (friendly messages)."""

from amortized.api.agent import _friendly_provider_error


def test_tool_calling_disabled() -> None:
    raw = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
    assert "tool-calling" in _friendly_provider_error(raw).lower()


def test_auth_by_name_and_text() -> None:
    assert "key" in _friendly_provider_error("", "ProviderAuthError").lower()
    assert "key" in _friendly_provider_error("Unauthorized: invalid api key").lower()


def test_model_not_found() -> None:
    assert "available on this provider" in _friendly_provider_error("model does not exist")


def test_rate_limited() -> None:
    assert "rate-limiting" in _friendly_provider_error("429 too many requests").lower()


def test_unmapped_falls_back_to_raw() -> None:
    assert _friendly_provider_error("some unusual provider error") == "some unusual provider error"


def test_empty_falls_back_to_generic() -> None:
    assert _friendly_provider_error("") == "The model provider returned an error."
