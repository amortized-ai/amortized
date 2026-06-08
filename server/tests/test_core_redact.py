"""Tests for config redaction utility."""

from amortized.core.redact import redact_config


class TestRedactConfig:
    def test_redacts_api_key(self) -> None:
        config = {"model": "openai/gpt-4o", "api_key": "sk-secret-123"}
        result = redact_config(config)
        assert result["api_key"] == "***redacted***"
        assert result["model"] == "openai/gpt-4o"

    def test_redacts_multiple_sensitive_fields(self) -> None:
        config = {"api_key": "sk-123", "api_secret": "sec", "password": "pw", "model": "x"}
        result = redact_config(config)
        assert result["api_key"] == "***redacted***"
        assert result["api_secret"] == "***redacted***"
        assert result["password"] == "***redacted***"
        assert result["model"] == "x"

    def test_empty_value_not_redacted(self) -> None:
        config = {"api_key": "", "model": "openai/gpt-4o"}
        result = redact_config(config)
        assert result["api_key"] == ""

    def test_none_value_not_redacted(self) -> None:
        config = {"api_key": None, "model": "openai/gpt-4o"}
        result = redact_config(config)
        assert result["api_key"] is None

    def test_no_sensitive_fields(self) -> None:
        config = {"model": "openai/gpt-4o", "num_samples": 100}
        result = redact_config(config)
        assert result == config

    def test_does_not_mutate_original(self) -> None:
        config = {"api_key": "sk-secret-123", "model": "x"}
        redact_config(config)
        assert config["api_key"] == "sk-secret-123"
