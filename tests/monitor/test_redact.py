"""Tests for value-only secret scrubbing (`monitor_redact.scrub_text`).

The monitor captures an allowlist of safe fields (no raw inputs, no key denylist),
so scrubbing only has to defend free-text/output against *pasted/echoed* secrets.
"""

from __future__ import annotations

from amortized.api.monitor_redact import _MAX_STR, _REDACTED, scrub_text


def test_scrubs_url_embedded_credentials() -> None:
    out = scrub_text("connect to https://user:s3cr3t@mlflow.svc:5000/api now")
    assert "s3cr3t" not in out
    assert f"https://{_REDACTED}@mlflow.svc:5000/api" in out


def test_scrubs_env_assignment() -> None:
    out = scrub_text("export OPENAI_API_KEY=sk-abcdef1234567890 && run")
    assert "sk-abcdef1234567890" not in out
    assert "OPENAI_API_KEY=" in out


def test_scrubs_bearer_jwt_provider_and_aws_keys() -> None:
    assert "abcdef0123456789" not in scrub_text("header Bearer abcdef0123456789 end")
    assert "eyJhbGciOi.eyJzdWIiOi.SflKxwRJSM" not in scrub_text(
        "t eyJhbGciOi.eyJzdWIiOi.SflKxwRJSM x"
    )
    assert "sk-ABCDEF0123456789" not in scrub_text("key sk-ABCDEF0123456789 leaked")
    assert "AKIAIOSFODNN7EXAMPLE" not in scrub_text("id AKIAIOSFODNN7EXAMPLE used")


def test_leaves_ordinary_text_untouched() -> None:
    # no key denylist -> ordinary words and config-ish values pass through
    assert scrub_text("Which base model? Qwen2.5-7B vs Llama-3-8B") == (
        "Which base model? Qwen2.5-7B vs Llama-3-8B"
    )
    assert scrub_text("max_tokens 2048, tokenizer Qwen/Qwen2.5") == (
        "max_tokens 2048, tokenizer Qwen/Qwen2.5"
    )


def test_truncates_long_strings() -> None:
    out = scrub_text("x" * (_MAX_STR + 500))
    assert out.endswith("chars]")
    assert len(out) < _MAX_STR + 500


def test_never_raises() -> None:
    assert scrub_text("") == ""
