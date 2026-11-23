"""Model-liveness checker.

Sends one minimal chat request per endpoint — the same calling paradigm the eval runner uses
(OpenAI ``/chat/completions``, or the Anthropic Messages API for ``provider_type`` anthropic) —
and classifies the response as passed, failed, or unverified. Each endpoint's whole attempt and
retry sequence runs under a hard wall-clock deadline (httpx timeouts only bound individual
network operations), so a slow endpoint can't outlive the caller's timeout.

Every probe is authorized by the destination policy (``destinations``) and connects only to the
vetted addresses; redirects are never followed and at most MAX_RESPONSE_BYTES of a reply is read.

Failure policy: ``failed`` (which blocks the job) is reserved for answers showing the
configuration can't work — a destination the policy denies, rejected credentials (401/403), an
unknown model route (404), a rejected request (other 4xx), or a reply that isn't a chat
response. Everything that only shows the endpoint didn't answer *this* probe is ``unverified``
(a warning): a timeout or the overall deadline (a cold, scale-to-zero endpoint can miss a 20 s
probe and still serve a job whose runner waits 300 s per request), 5xx after retries, 429, a
host that didn't resolve, and connection failures.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from typing import Any

import httpx

from amortized.model_check import destinations
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
# Probes in flight at once in this process. Beyond it work is refused, not queued: a queued probe
# would burn its caller's deadline.
MAX_INFLIGHT_PROBES = 32
# A 16-token chat reply is tiny; anything larger is refused rather than buffered.
MAX_RESPONSE_BYTES = 64 * 1024

_inflight = 0


def try_reserve(count: int) -> bool:
    """Reserve ``count`` probe slots under MAX_INFLIGHT_PROBES; False when they don't fit.
    Pair a True with ``release(count)``. (asyncio is single-threaded, so no lock is needed.)"""
    global _inflight
    if _inflight + count > MAX_INFLIGHT_PROBES:
        return False
    _inflight += count
    return True


def release(count: int) -> None:
    global _inflight
    _inflight = max(0, _inflight - count)

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
    url = destinations.chat_request_url(ep.base_url, ep.provider_type)
    messages = [{"role": "user", "content": "ping"}]
    headers = dict(destinations.IDENTITY_ENCODING)
    if ep.provider_type == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if ep.api_key:
            headers["x-api-key"] = ep.api_key
        return (
            url,
            headers,
            {"model": ep.model, "max_tokens": PROBE_MAX_TOKENS, "messages": messages},
        )
    if ep.api_key:
        headers["Authorization"] = f"Bearer {ep.api_key}"
    return url, headers, {"model": ep.model, "messages": messages, "max_tokens": PROBE_MAX_TOKENS}


def _valid_shape(provider_type: str, data: Any) -> bool:
    """A 2xx only proves something answered; require the provider's response shape too."""
    if not isinstance(data, dict):
        return False
    if provider_type == "anthropic":
        return data.get("type") == "message" and isinstance(data.get("content"), list)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    choice = choices[0]
    # A reasoning model can spend the whole probe budget thinking and reply with
    # message: null and finish_reason: length. The model still answered.
    if isinstance(choice.get("message"), dict):
        return True
    return choice.get("message") is None and bool(choice.get("finish_reason"))


def _status_text(resp: httpx.Response) -> str:
    """The numeric status only. Details reach the agent (Morty) and the UI, so they're built
    from the status and local reason codes — never the provider's reason phrase or body."""
    return f"HTTP {resp.status_code}"


async def _probe(ep: ModelEndpoint, deadline_at: float, approved: tuple[str, ...]) -> CheckResult:
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
    # Connect only to the addresses authorized here, so DNS can't be rebound in between.
    decision = await destinations.authorize(url, approved)
    if decision.verdict == destinations.DENIED:
        return result(FAILED, "destination_denied", decision.reason)
    if decision.verdict == destinations.UNRESOLVED:
        return result(UNVERIFIED, "dns_unresolved", decision.reason)
    loop = asyncio.get_running_loop()
    adapted = False
    last = result(UNVERIFIED, "unreachable", "no attempt completed")
    attempt = 0
    async with httpx.AsyncClient(
        transport=destinations.pinned_transport(decision), follow_redirects=False
    ) as client:
        while attempt < MAX_ATTEMPTS:
            timeout = max(0.1, min(ATTEMPT_TIMEOUT_SECONDS, deadline_at - loop.time()))
            try:
                async with client.stream(
                    "POST", url, json=body, headers=headers, timeout=timeout
                ) as resp:
                    # Only a 2xx (shape check) or 400 (token-param adaptation) needs the body:
                    # every other status is classified from the status line, so a stalled body
                    # can't turn a received 401/404 into a timeout.
                    raw, truncated = b"", False
                    if 200 <= resp.status_code < 300 or resp.status_code == 400:
                        raw, truncated = await destinations.read_limited(resp, MAX_RESPONSE_BYTES)
            except httpx.InvalidURL:
                return result(FAILED, "invalid_url", "invalid URL")
            # Error text can carry peer-supplied content (e.g. a TLS alert); the class is enough.
            except httpx.TimeoutException as exc:
                last = result(UNVERIFIED, "deadline", type(exc).__name__)
            except httpx.TransportError as exc:  # connect/DNS/protocol failures
                last = result(UNVERIFIED, "unreachable", type(exc).__name__)
            else:
                status = resp.status_code
                text = raw.decode("utf-8", errors="replace")
                # Reasoning models reject max_tokens; the eval runner adapts the same way.
                if (
                    status == 400
                    and not adapted
                    and "max_tokens" in body
                    and "max_completion_tokens" in text
                ):
                    body = {k: v for k, v in body.items() if k != "max_tokens"}
                    body["max_completion_tokens"] = PROBE_MAX_TOKENS
                    adapted = True
                    continue
                if 200 <= status < 300:
                    if truncated:
                        detail = f"response larger than {MAX_RESPONSE_BYTES} bytes"
                        return result(FAILED, "bad_response", detail)
                    try:
                        data = json.loads(raw)
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
                if status == 408:  # the endpoint timed out the request: transient
                    last = result(UNVERIFIED, "deadline", _status_text(resp))
                elif status < 500:
                    return result(FAILED, "bad_request", _status_text(resp))
                else:
                    last = result(UNVERIFIED, "server_error", _status_text(resp))
            if attempt + 1 < MAX_ATTEMPTS:
                await asyncio.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
            attempt += 1
    return last


async def check_endpoint(
    ep: ModelEndpoint, deadline: float, approved: tuple[str, ...] = ()
) -> CheckResult:
    """Check one endpoint within ``deadline`` seconds of wall-clock time; always returns a
    result. ``approved`` are the operator-approved URL prefixes for private services."""
    loop = asyncio.get_running_loop()
    # Each endpoint runs in its own task (check_endpoints gathers them), so this is per-probe.
    token = _probe_secret.set(ep.api_key)
    try:
        async with asyncio.timeout(deadline):
            return await _probe(ep, loop.time() + deadline, approved)
    except TimeoutError:
        return CheckResult(
            ep.label,
            ep.model,
            redact_url(ep.base_url),
            UNVERIFIED,
            "deadline",
            f"no answer within {deadline:.0f}s"
            if deadline >= 1
            else f"no answer within {deadline:.1f}s",
        )
    except Exception as exc:  # a checker bug must not drop this or any other endpoint's result
        logger.warning("model check internal error for %s (%s)", ep.label, type(exc).__name__)
        # Unverified, not failed: a bug in the checker says nothing about the endpoint, so it
        # must not block the job (checker problems are never definite failures).
        return CheckResult(
            ep.label,
            ep.model,
            redact_url(ep.base_url),
            UNVERIFIED,
            "internal_error",
            type(exc).__name__,
        )
    finally:
        _probe_secret.reset(token)


async def check_endpoints(
    endpoints: list[ModelEndpoint], deadline: float, approved: tuple[str, ...] = ()
) -> list[CheckResult]:
    """Check all endpoints concurrently, each under its own deadline; results keep input order."""
    return list(
        await asyncio.gather(*(check_endpoint(ep, deadline, approved) for ep in endpoints))
    )
