"""Model-liveness checker and service: outcomes, deadlines, redaction, and bounds."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable

import httpx
import pytest

from amortized.core.model_endpoints import FAILED, PASSED, UNVERIFIED, ModelEndpoint
from amortized.model_check import checker, service
from amortized.model_check.types import CheckResult

Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]

OPENAI_OK = {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}
ANTHROPIC_OK = {"type": "message", "content": [{"type": "text", "text": "pong"}]}


def _ep(label: str = "judge", base_url: str = "http://gw:5000/v1", **kw: str) -> ModelEndpoint:
    return ModelEndpoint(label=label, base_url=base_url, model=kw.pop("model", "gpt-oss"), **kw)


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    monkeypatch.setattr(checker, "BACKOFF_SECONDS", (0.0, 0.0))
    return []


def _route(monkeypatch: pytest.MonkeyPatch, handler: Handler, seen: list[httpx.Request]) -> None:
    real_client = httpx.AsyncClient

    async def _handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        response = handler(request)
        if asyncio.iscoroutine(response):
            response = await response
        return response  # type: ignore[return-value]

    monkeypatch.setattr(
        checker.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(_handle), **kw),
    )


def _check(ep: ModelEndpoint, deadline: float = 5.0):  # type: ignore[no-untyped-def]
    return asyncio.run(checker.check_endpoint(ep, deadline))


class TestOutcomes:
    def test_openai_endpoint_passes(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(200, json=OPENAI_OK), seen)
        result = _check(_ep(api_key="sk-1", credential_id="K:abc"))
        assert (result.outcome, result.reason) == (PASSED, "ok")
        (req,) = seen
        assert str(req.url) == "http://gw:5000/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer sk-1"
        body = json.loads(req.content)
        assert body == {
            "model": "gpt-oss",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
        }

    def test_unauthenticated_endpoint_sends_no_auth_header(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(200, json=OPENAI_OK), seen)
        assert _check(_ep()).outcome == PASSED
        assert "Authorization" not in seen[0].headers

    def test_reasoning_model_token_param_is_adapted(self, monkeypatch, seen) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "max_tokens" in json.loads(request.content):
                return httpx.Response(400, text="use 'max_completion_tokens' instead")
            return httpx.Response(200, json=OPENAI_OK)

        _route(monkeypatch, handler, seen)
        assert _check(_ep()).outcome == PASSED
        assert json.loads(seen[1].content)["max_completion_tokens"] == 16

    def test_not_found(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(404, text="no such route"), seen)
        result = _check(_ep())
        assert (result.outcome, result.reason) == (FAILED, "not_found")
        assert "404" in result.detail
        assert "no such route" not in result.detail  # httpx status text only, no provider body

    def test_auth_required_vs_auth_failed(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(401), seen)
        no_key = _check(_ep())
        with_key = _check(_ep(api_key="sk-bad", credential_id="K:x"))
        assert (no_key.outcome, no_key.reason) == (FAILED, "auth_required")
        assert "requires an API key" in no_key.detail
        assert (with_key.outcome, with_key.reason) == (FAILED, "auth_failed")

    def test_connection_refused_is_unreachable_after_retries(self, monkeypatch, seen) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        _route(monkeypatch, handler, seen)
        result = _check(_ep())
        assert (result.outcome, result.reason) == (FAILED, "unreachable")
        assert len(seen) == checker.MAX_ATTEMPTS

    def test_persistent_5xx_is_server_error(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(502), seen)
        result = _check(_ep())
        assert (result.outcome, result.reason) == (FAILED, "server_error")
        assert len(seen) == checker.MAX_ATTEMPTS

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, json={}),
            httpx.Response(200, json={"choices": []}),
            httpx.Response(200, text="<html>login</html>"),
            httpx.Response(200, json={"choices": [{"text": "legacy"}]}),
        ],
    )
    def test_malformed_2xx_is_bad_response(self, monkeypatch, seen, response) -> None:
        _route(monkeypatch, lambda _r: response, seen)
        result = _check(_ep())
        assert (result.outcome, result.reason) == (FAILED, "bad_response")

    def test_rate_limited_is_unverified(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(429), seen)
        result = _check(_ep())
        assert (result.outcome, result.reason) == (UNVERIFIED, "rate_limited")

    def test_anthropic_messages_api(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(200, json=ANTHROPIC_OK), seen)
        ep = _ep(
            base_url="https://api.anthropic.com/v1",
            provider_type="anthropic",
            api_key="ak-1",
            credential_id="A:x",
        )
        assert _check(ep).outcome == PASSED
        (req,) = seen
        assert str(req.url) == "https://api.anthropic.com/v1/messages"
        assert req.headers["x-api-key"] == "ak-1"
        assert "Authorization" not in req.headers

    def test_anthropic_requires_its_response_shape(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(200, json=OPENAI_OK), seen)
        ep = _ep(base_url="https://api.anthropic.com/v1", provider_type="anthropic")
        assert _check(ep).reason == "bad_response"


class TestDeadlines:
    def test_slow_endpoint_fails_within_deadline_while_sibling_passes(
        self, monkeypatch, seen
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "slow":
                await asyncio.sleep(10)
            return httpx.Response(200, json=OPENAI_OK)

        _route(monkeypatch, handler, seen)
        started = time.monotonic()
        slow, fast = asyncio.run(
            checker.check_endpoints(
                [_ep("slow judge", "http://slow/v1"), _ep("fast judge", "http://fast/v1")], 0.5
            )
        )
        assert time.monotonic() - started < 2.0
        assert (slow.outcome, slow.reason) == (FAILED, "deadline")
        assert (fast.outcome, fast.reason) == (PASSED, "ok")

    def test_internal_error_keeps_other_results(self, monkeypatch, seen) -> None:
        _route(monkeypatch, lambda _r: httpx.Response(200, json=OPENAI_OK), seen)
        real = checker._request_for

        def flaky(ep: ModelEndpoint):  # type: ignore[no-untyped-def]
            if ep.label == "broken":
                raise RuntimeError("bug")
            return real(ep)

        monkeypatch.setattr(checker, "_request_for", flaky)
        broken, ok = asyncio.run(checker.check_endpoints([_ep("broken"), _ep("fine")], 5.0))
        assert (broken.outcome, broken.reason) == (FAILED, "internal_error")
        assert ok.outcome == PASSED


class TestService:
    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.app), base_url="http://svc"
        )

    def _post(self, payload: dict) -> httpx.Response:  # type: ignore[type-arg]
        async def go() -> httpx.Response:
            async with self._client() as c:
                return await c.post("/check", json=payload)

        return asyncio.run(go())

    def test_check_returns_results_without_keys(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        async def fake(endpoints, deadline):  # type: ignore[no-untyped-def]
            captured["deadline"] = deadline
            captured["keys"] = [e.api_key for e in endpoints]
            return [CheckResult(e.label, e.model, e.base_url, PASSED, "ok") for e in endpoints]

        monkeypatch.setattr(checker, "check_endpoints", fake)
        resp = self._post(
            {
                "deadline_seconds": 99,
                "endpoints": [
                    {
                        "label": "judge",
                        "base_url": "http://gw/v1",
                        "model": "m",
                        "api_key": "sk-secret",
                    }
                ],
            }
        )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["outcome"] == PASSED
        assert "sk-secret" not in resp.text
        assert captured["keys"] == ["sk-secret"]
        assert captured["deadline"] == service.MAX_DEADLINE_SECONDS  # clamped

    def test_too_many_endpoints_is_413(self) -> None:
        endpoints = [
            {"label": f"e{i}", "base_url": "http://gw/v1", "model": "m"}
            for i in range(service.MAX_ENDPOINTS_PER_REQUEST + 1)
        ]
        assert self._post({"endpoints": endpoints}).status_code == 413

    def test_over_inflight_cap_is_503(self, monkeypatch) -> None:
        monkeypatch.setattr(service, "_inflight", service.MAX_INFLIGHT_PROBES)
        resp = self._post({"endpoints": [{"label": "e", "base_url": "http://gw/v1", "model": "m"}]})
        assert resp.status_code == 503

    def test_validation_error_does_not_echo_input(self) -> None:
        resp = self._post(
            {
                "endpoints": [
                    {
                        "label": "e",
                        "base_url": "http://gw/v1",
                        "model": "m",
                        "provider_type": "nope",
                        "api_key": "sk-leak",
                    }
                ]
            }
        )
        assert resp.status_code == 422
        assert "sk-leak" not in resp.text

    def test_healthz(self) -> None:
        async def go() -> httpx.Response:
            async with self._client() as c:
                return await c.get("/healthz")

        assert asyncio.run(go()).status_code == 200
