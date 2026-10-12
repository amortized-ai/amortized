"""Tests for provider-error mapping surfaced in Morty chat (friendly messages)."""

from amortized.api.agent import _classify_provider_error, _friendly_provider_error


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


def test_classify_returns_stable_codes() -> None:
    raw_tool = "tool choice requires --enable-auto-tool-choice and --tool-call-parser"
    assert _classify_provider_error(raw_tool)[0] == "tool_calling"
    assert _classify_provider_error("", "ProviderAuthError")[0] == "auth"
    assert _classify_provider_error("Unauthorized: invalid api key")[0] == "auth"
    assert _classify_provider_error("model does not exist")[0] == "model_unavailable"
    assert _classify_provider_error("", "ProviderModelNotFoundError")[0] == "model_unavailable"
    assert _classify_provider_error("429 too many requests")[0] == "rate_limit"
    assert _classify_provider_error("some unusual provider error")[0] == "unknown"


def test_structured_name_wins_over_message_text() -> None:
    # A structured ProviderAuthError stays auth even when its message mentions "not found".
    assert _classify_provider_error("API key not found", "ProviderAuthError")[0] == "auth"


def test_text_only_prefers_missing_over_auth() -> None:
    # Without a structured name, the more specific missing-model signal wins over "permission".
    assert _classify_provider_error("permission denied: model not found")[0] == "model_unavailable"


def test_classify_message_matches_shim() -> None:
    # The back-compat shim returns exactly the classifier's message half.
    for raw, name in [("model does not exist", ""), ("", "ProviderAuthError"), ("", "")]:
        assert _friendly_provider_error(raw, name) == _classify_provider_error(raw, name)[1]
