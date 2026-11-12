"""Model-liveness checker.

Sends one minimal chat request per endpoint — the same calling paradigm the eval runner uses
(OpenAI ``/chat/completions``, or the Anthropic Messages API for ``provider_type`` anthropic) —
and classifies the response as passed, failed, or unverified. Each endpoint's whole attempt and
retry sequence runs under a hard wall-clock deadline (httpx timeouts only bound individual
network operations), so a slow endpoint becomes a ``failed/deadline`` result instead of
outliving the caller's timeout.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any

import httpx

from amortized.model_check.types import (
    FAILED,
    PASSED,
    UNVERIFIED,
    CheckResult,
    ModelEndpoint,
    redact,
    redact_url,
)

logger = logging.getLogger("amortized.model_check")

ANTHROPIC_VERSION = "2023-06-01"
PROBE_MAX_TOKENS = 16
ATTEMPT_TIMEOUT_SECONDS = 10.0
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (0.5, 1.0)

# httpx logs every request at INFO with the full URL ("HTTP Request: POST <url> ..."), and a
# probed URL can carry userinfo or query credentials. While a probe runs, the key being probed is
# held here; the filter below redacts httpx's log lines for that probe — in the standalone service
# and in-process in the server — without touching httpx's logging for anything else.
_probe_secret: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "model_check_probe_secret", default=None
)


class _RedactProbeRequestLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        secret = _probe_secret.get()
        if secret is not None:
            record.msg = redact(record.getMessage(), [secret])
            record.args = None
        return True


_httpx_logger = logging.getLogger("httpx")
if not any(isinstance(f, _RedactProbeRequestLogs) for f in _httpx_logger.filters):
    _httpx_logger.addFilter(_RedactProbeRequestLogs())


def _request_for(ep: ModelEndpoint) -> tuple[str, dict[str, str], dict[str, Any]]:
    base = ep.base_url.rstrip("/")
    messages = [{"role": "user", "content": "ping"}]
    if ep.provider_type == "anthropic":
        url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if ep.api_key:
            headers["x-api-key"] = ep.api_key
        return (
            url,
            headers,
            {"model": ep.model, "max_tokens": PROBE_MAX_TOKENS, "messages": messages},
        )
    headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
    url = base + "/chat/completions"
    return url, headers, {"model": ep.model, "messages": messages, "max_tokens": PROBE_MAX_TOKENS}


def _valid_shape(provider_type: str, data: Any) -> bool:
    """A 2xx only proves something answered; require the provider's response shape too."""
    if not isinstance(data, dict):
        return False
    if provider_type == "anthropic":
        return data.get("type") == "message" and isinstance(data.get("content"), list)
    choices = data.get("choices")
    return (
        isinstance(choices, list)
        and bool(choices)
        and isinstance(choices[0], dict)
        and isinstance(choices[0].get("message"), dict)
    )


def _status_text(resp: httpx.Response) -> str:
    """httpx's own status-line text (no provider body), e.g. "Client error '404 Not Found'
    for url '…'"."""
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return str(exc).splitlines()[0]
    return f"HTTP {resp.status_code}"


async def _probe(ep: ModelEndpoint, deadline_at: float) -> CheckResult:
    def result(outcome: str, reason: str, detail: str = "") -> CheckResult:
        return CheckResult(
            ep.label,
            ep.model,
            redact_url(ep.base_url),
            outcome,
            reason,
            redact(detail, [ep.api_key]),
        )

    url, headers, body = _request_for(ep)
    loop = asyncio.get_running_loop()
    adapted = False
    last = result(FAILED, "unreachable", "no attempt completed")
    attempt = 0
    async with httpx.AsyncClient(follow_redirects=False) as client:
        while attempt < MAX_ATTEMPTS:
            timeout = max(0.1, min(ATTEMPT_TIMEOUT_SECONDS, deadline_at - loop.time()))
            try:
                resp = await client.post(url, json=body, headers=headers, timeout=timeout)
            except httpx.InvalidURL as exc:
                return result(FAILED, "unreachable", f"invalid URL: {exc}")
            except (
                httpx.TransportError
            ) as exc:  # connect/DNS/read failures and per-attempt timeouts
                last = result(FAILED, "unreachable", str(exc) or type(exc).__name__)
            else:
                status = resp.status_code
                # Reasoning models reject max_tokens; the eval runner adapts the same way.
                if (
                    status == 400
                    and not adapted
                    and "max_tokens" in body
                    and "max_completion_tokens" in resp.text
                ):
                    body = {k: v for k, v in body.items() if k != "max_tokens"}
                    body["max_completion_tokens"] = PROBE_MAX_TOKENS
                    adapted = True
                    continue
                if 200 <= status < 300:
                    try:
                        data = resp.json()
                    except ValueError:
                        data = None
                    if _valid_shape(ep.provider_type, data):
                        return result(PASSED, "ok")
                    return result(
                        FAILED, "bad_response", f"unexpected response shape (HTTP {status})"
                    )
                if status == 429:
                    return result(UNVERIFIED, "rate_limited", _status_text(resp))
                if status in (401, 403):
                    reason = "auth_failed" if ep.api_key else "auth_required"
                    detail = _status_text(resp)
                    if not ep.api_key:
                        detail += " — the endpoint requires an API key"
                    return result(FAILED, reason, detail)
                if status == 404:
                    return result(FAILED, "not_found", _status_text(resp))
                if status < 500:
                    return result(FAILED, "bad_request", _status_text(resp))
                last = result(FAILED, "server_error", _status_text(resp))
            if attempt + 1 < MAX_ATTEMPTS:
                await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
            attempt += 1
    return last


async def check_endpoint(ep: ModelEndpoint, deadline: float) -> CheckResult:
    """Check one endpoint within ``deadline`` seconds of wall-clock time; always returns a
    result."""
    loop = asyncio.get_running_loop()
    # Each endpoint runs in its own task (check_endpoints gathers them), so this is per-probe.
    token = _probe_secret.set(ep.api_key)
    try:
        async with asyncio.timeout(deadline):
            return await _probe(ep, loop.time() + deadline)
    except TimeoutError:
        return CheckResult(
            ep.label,
            ep.model,
            redact_url(ep.base_url),
            FAILED,
            "deadline",
            f"no answer within {deadline:g}s",
        )
    except Exception as exc:  # a checker bug must not drop this or any other endpoint's result
        logger.warning("model check internal error for %s (%s)", ep.label, type(exc).__name__)
        return CheckResult(
            ep.label,
            ep.model,
            redact_url(ep.base_url),
            FAILED,
            "internal_error",
            type(exc).__name__,
        )
    finally:
        _probe_secret.reset(token)


async def check_endpoints(endpoints: list[ModelEndpoint], deadline: float) -> list[CheckResult]:
    """Check all endpoints concurrently, each under its own deadline; results keep input order."""
    return list(await asyncio.gather(*(check_endpoint(ep, deadline) for ep in endpoints)))
