"""Model-check client and builder hooks: budgets, chunking, checker outages,
credential resolution parity, and which endpoints each job type declares."""

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

import amortized.config as config_mod
from amortized.core import model_check_client as client
from amortized.core import model_endpoints
from amortized.core.job_env import forwarded_env
from amortized.core.model_endpoints import (
    FAILED,
    PASSED,
    UNVERIFIED,
    CheckResult,
    ConfigError,
    ModelEndpoint,
)
from amortized.jobs import eval as eval_builder
from amortized.jobs import sdg as sdg_builder
from amortized.model_check import checker


def _eps(n: int) -> list[ModelEndpoint]:
    return [ModelEndpoint(label=f"e{i}", base_url=f"http://h{i}/v1", model="m") for i in range(n)]


def _ok(ep: ModelEndpoint) -> CheckResult:
    return CheckResult(ep.label, ep.model, ep.base_url, PASSED, "ok")


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Point the client at a fake check service; returns a list the handler appends to."""
    monkeypatch.setattr(config_mod.settings, "model_check_url", "http://svc:7351")
    calls: list[dict[str, Any]] = []
    real_client = httpx.AsyncClient

    def install(handler, timeouts: list[Any] | None = None) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
        async def _handle(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            calls.append(payload)
            result = handler(payload)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        def factory(**kw: Any) -> httpx.AsyncClient:
            if timeouts is not None:
                timeouts.append(kw.get("timeout"))
            return real_client(transport=httpx.MockTransport(_handle), **kw)

        monkeypatch.setattr(client.httpx, "AsyncClient", factory)
        return calls

    return install


def _results_for(payload: dict[str, Any], outcome: str = PASSED) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "results": [
                {
                    "label": e["label"],
                    "model": e["model"],
                    "base_url": e["base_url"],
                    "outcome": outcome,
                    "reason": "ok" if outcome == PASSED else "x",
                    "detail": "",
                }
                for e in payload["endpoints"]
            ]
        },
    )


class TestCheckerAvailability:
    """Checker problems are unverified — never passed, never failed."""

    @pytest.mark.parametrize(
        ("handler", "reason"),
        [
            (lambda p: (_ for _ in ()).throw(httpx.ConnectError("refused")), "checker_unavailable"),
            (lambda p: (_ for _ in ()).throw(httpx.ReadTimeout("slow")), "checker_unavailable"),
            (lambda p: httpx.Response(503), "checker_busy"),
            (lambda p: httpx.Response(500), "checker_unavailable"),
            (lambda p: httpx.Response(200, text="not json"), "checker_unavailable"),
            (lambda p: httpx.Response(200, json={"results": "nope"}), "checker_unavailable"),
        ],
    )
    def test_outage_is_unverified(self, service, handler, reason) -> None:
        service(handler)
        results = asyncio.run(client.run_checks(_eps(2), 5.0))
        assert [(r.outcome, r.reason) for r in results] == [(UNVERIFIED, reason)] * 2

    def test_partial_or_mismatched_results_yield_one_result_each(self, service) -> None:
        def handler(payload: dict[str, Any]) -> httpx.Response:
            good = _results_for(payload).json()["results"]
            good[1]["label"] = "someone-else"  # mismatched identity
            return httpx.Response(200, json={"results": good[:2]})  # and one missing

        service(handler)
        results = asyncio.run(client.run_checks(_eps(3), 5.0))
        assert [r.label for r in results] == ["e0", "e1", "e2"]
        assert [r.outcome for r in results] == [PASSED, UNVERIFIED, UNVERIFIED]

    def test_http_timeout_fits_the_budget_and_exceeds_the_probe_deadline(self, service) -> None:
        timeouts: list[Any] = []
        calls = service(_results_for, timeouts)
        asyncio.run(client.run_checks(_eps(1), 7.0))
        assert calls[0]["deadline_seconds"] < timeouts[0] <= 7.0

    def test_stalled_checker_returns_within_the_total_budget(self, service) -> None:
        async def stall(payload: dict[str, Any]) -> httpx.Response:
            await asyncio.sleep(10)
            return _results_for(payload)

        service(stall)
        started = time.monotonic()
        results = asyncio.run(client.run_checks(_eps(3), 1.0))
        assert time.monotonic() - started < 1.6
        assert [r.outcome for r in results] == [UNVERIFIED] * 3

    def test_hung_check_is_cancelled_at_the_budget(self, monkeypatch) -> None:
        monkeypatch.setattr(config_mod.settings, "model_check_url", "")

        async def hang(endpoints, deadline):  # type: ignore[no-untyped-def]
            await asyncio.sleep(100)

        monkeypatch.setattr("amortized.model_check.checker.check_endpoints", hang)
        started = time.monotonic()
        results = asyncio.run(client.run_checks(_eps(2), 0.5))
        assert time.monotonic() - started < 1.0
        assert [(r.outcome, r.reason) for r in results] == [(UNVERIFIED, "budget_exhausted")] * 2

    def test_service_definite_failure_passes_through(self, service) -> None:
        service(lambda p: _results_for(p, FAILED))
        (result,) = asyncio.run(client.run_checks(_eps(1), 5.0))
        assert result.outcome == FAILED


class TestChunkingAndBudget:
    def test_nine_endpoints_two_requests_one_result_each(self, service) -> None:
        calls = service(_results_for)
        results = asyncio.run(client.run_checks(_eps(9), 5.0))
        assert sorted(len(c["endpoints"]) for c in calls) == [1, 8]
        assert [r.label for r in results] == [f"e{i}" for i in range(9)]
        assert all(r.outcome == PASSED for r in results)

    def test_cancelling_the_check_cancels_every_chunk(self, monkeypatch) -> None:
        # asyncio.wait doesn't cancel its tasks: if the caller is cancelled, running chunks
        # must be cancelled and queued ones must never start.
        started: list[int] = []
        cancelled: list[int] = []

        async def slow_chunk(chunk, deadline, timeout):  # type: ignore[no-untyped-def]
            started.append(len(chunk))
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(len(chunk))
                raise
            return [_ok(e) for e in chunk]

        monkeypatch.setattr(client, "_check_chunk", slow_chunk)

        async def go() -> None:
            task = asyncio.create_task(client.run_checks(_eps(17), 20.0))
            await asyncio.sleep(0.05)  # the first wave (two chunks) is running
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.05)  # a leaked queued chunk would start here

        asyncio.run(go())
        assert len(started) == client.MAX_CONCURRENT_CHUNKS  # the third chunk never started
        assert sorted(cancelled) == sorted(started)

    def test_seventeen_endpoints_at_most_two_chunks_in_flight(self, service) -> None:
        in_flight = 0
        peak = 0

        async def handler(payload: dict[str, Any]) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return _results_for(payload)

        calls = service(handler)
        results = asyncio.run(client.run_checks(_eps(17), 5.0))
        assert len(calls) == 3
        assert peak <= client.MAX_CONCURRENT_CHUNKS
        assert [r.label for r in results] == [f"e{i}" for i in range(17)]

    def test_chunk_that_cannot_start_in_budget_is_budget_exhausted(self, service) -> None:
        async def handler(payload: dict[str, Any]) -> httpx.Response:
            await asyncio.sleep(1.0)
            return _results_for(payload)

        service(handler)
        started = time.monotonic()
        results = asyncio.run(client.run_checks(_eps(17), 1.5))
        assert time.monotonic() - started < 2.5  # one total budget, not one per chunk
        assert [r.outcome for r in results[:16]] == [PASSED] * 16
        assert (results[16].outcome, results[16].reason) == (UNVERIFIED, "budget_exhausted")

    def test_first_chunk_always_runs_even_with_a_tiny_budget(self, service) -> None:
        # Regression: a budget below MIN_CHUNK_BUDGET_SECONDS used to mark even the first chunk
        # budget_exhausted (unverified) — skipping the check entirely.
        calls = service(_results_for)
        results = asyncio.run(client.run_checks(_eps(2), 0.3))
        assert len(calls) == 1
        assert calls[0]["deadline_seconds"] <= 0.3
        assert [r.outcome for r in results] == [PASSED, PASSED]

    def test_in_process_when_no_service_url(self, monkeypatch) -> None:
        monkeypatch.setattr(config_mod.settings, "model_check_url", "")
        seen: list[float] = []

        async def fake(endpoints, deadline):  # type: ignore[no-untyped-def]
            seen.append(deadline)
            return [_ok(e) for e in endpoints]

        monkeypatch.setattr("amortized.model_check.checker.check_endpoints", fake)
        results = asyncio.run(client.run_checks(_eps(3), 4.0))
        assert len(results) == 3 and seen and seen[0] <= 4.0


class TestCheckJobModels:
    def test_job_type_without_hook_has_nothing_to_check(self) -> None:
        report = asyncio.run(client.check_job_models("training", {}))
        assert report.config_errors == [] and report.results == []

    def test_config_errors_skip_probing(self, monkeypatch) -> None:
        async def boom(*_a, **_k):  # type: ignore[no-untyped-def]
            raise AssertionError("must not probe when the config is wrong")

        monkeypatch.setattr(client, "run_checks", boom)
        config = {
            "judge": {"base_url": "not-a-provider", "model": "m"},
            "rubric": [{"name": "x"}],
            "endpoint": {"base_url": "http://tuned/v1", "model": "t"},
        }
        report = asyncio.run(client.check_job_models("eval", config))
        assert [e.reason for e in report.config_errors] == ["invalid_base_url"]

    def test_dedup_keeps_distinct_credentials(self, monkeypatch) -> None:
        checked: list[ModelEndpoint] = []

        async def fake(endpoints, budget):  # type: ignore[no-untyped-def]
            checked.extend(endpoints)
            return [_ok(e) for e in endpoints]

        monkeypatch.setattr(client, "run_checks", fake)
        # Two providers serving the same URL and model under different keys.
        providers = {
            name: {
                "name": name,
                "endpoint": "http://gw/v1",
                "provider_type": "openai",
                "api_key": f"{name.upper()}_KEY",
            }
            for name in ("p_one", "p_two")
        }
        monkeypatch.setattr(eval_builder, "resolve_provider", lambda n: providers.get(n))
        monkeypatch.setattr(
            eval_builder,
            "inject_enabled_provider_keys",
            lambda env: env.update({"P_ONE_KEY": "k1", "P_TWO_KEY": "k2"}),
        )
        config = {
            "endpoint": {"base_url": "p_one", "model": "m"},
            "judge": {"base_url": "p_two", "model": "m"},
            "rubric": [{"name": "x"}],
        }
        asyncio.run(client.check_job_models("eval", config))
        assert len(checked) == 2  # same model, different credentials -> both checked


class TestEvalHook:
    def _checks(self, config: dict[str, Any]):  # type: ignore[no-untyped-def]
        return eval_builder.model_checks(config, forwarded_env())

    def test_literal_url_judge_never_gets_a_forwarded_key(self, monkeypatch) -> None:
        # A literal URL is keyless (keyed models come from providers in Settings): a key the
        # server forwards to jobs is never sent to a URL the config names — not by the job,
        # and not by the check, which runs at validate time, before the user confirms.
        monkeypatch.setattr(config_mod.settings, "forward_env", ["EVAL_JUDGE_API_KEY"])
        monkeypatch.setenv("EVAL_JUDGE_API_KEY", "sk-forwarded")
        monkeypatch.setattr(eval_builder, "resolve_provider", lambda _n: None)
        endpoints, errors = self._checks(
            {
                "model_name_or_path": "Qwen/Qwen3.5-4B",
                "judge": {"base_url": "http://judge:8000/v1", "model": "gpt-oss"},
            }
        )
        assert errors == []
        (judge,) = endpoints
        assert judge.api_key == "" and judge.credential_id == ""

    def test_incomplete_endpoint_is_a_config_error(self, monkeypatch) -> None:
        # A retry's overrides skip request validation; build() would reject this judge at
        # dispatch, so the check rejects it up front instead of skipping it.
        monkeypatch.setattr(eval_builder, "resolve_provider", lambda _n: None)
        endpoints, errors = self._checks(
            {"model_name_or_path": "m", "judge": {"base_url": "http://judge/v1"}}
        )
        assert endpoints == []
        assert [(e.label, e.reason) for e in errors] == [("judge", "invalid_endpoint")]
        assert "base_url and model are required" in errors[0].message

    def test_classification_ignores_an_incomplete_judge(self) -> None:
        # Classification evals never call a judge (build() ignores it too).
        config = {
            "eval_mode": "classification",
            "training_job_id": "t",
            "judge": {"base_url": "http://judge/v1"},
        }
        assert self._checks(config) == ([], [])

    def test_unauthenticated_literal_url(self, monkeypatch) -> None:
        monkeypatch.setattr(config_mod.settings, "forward_env", [])
        monkeypatch.delenv("EVAL_JUDGE_API_KEY", raising=False)
        monkeypatch.setattr(eval_builder, "resolve_provider", lambda _n: None)
        endpoints, errors = self._checks(
            {"model_name_or_path": "m", "judge": {"base_url": "http://judge/v1", "model": "j"}}
        )
        assert errors == []
        (judge,) = endpoints
        assert judge.api_key == "" and judge.credential_id == ""

    def test_inline_key_is_a_config_error_never_probed(self, monkeypatch) -> None:
        # The request schema rejects inline keys; the hook still refuses one defensively.
        monkeypatch.setattr(eval_builder, "resolve_provider", lambda _n: None)
        endpoints, errors = self._checks(
            {
                "model_name_or_path": "m",
                "judge": {"base_url": "http://judge/v1", "model": "j", "api_key": "sk-x"},
            }
        )
        assert endpoints == []
        assert [(e.label, e.reason) for e in errors] == [("judge", "invalid_endpoint")]
        assert "sk-x" not in errors[0].message

    def test_provider_backed_judge(self, monkeypatch) -> None:
        provider = {
            "name": "openai",
            "endpoint": "https://api.openai.com/v1",
            "provider_type": "openai",
            "api_key": "OPENAI_API_KEY",
        }
        monkeypatch.setattr(
            eval_builder, "resolve_provider", lambda n: provider if n == "openai" else None
        )
        monkeypatch.setattr(
            eval_builder,
            "inject_enabled_provider_keys",
            lambda env: env.update({"OPENAI_API_KEY": "sk-provider"}),
        )
        endpoints, _ = self._checks(
            {"model_name_or_path": "m", "judge": {"base_url": "openai", "model": "gpt-5"}}
        )
        (judge,) = endpoints
        assert judge.base_url == "https://api.openai.com/v1"
        assert judge.api_key == "sk-provider"

    def test_embedded_model_and_classification_not_checked(self, monkeypatch) -> None:
        monkeypatch.setattr(eval_builder, "resolve_provider", lambda _n: None)
        endpoints, _ = self._checks(
            {"training_job_id": "tj", "judge": {"base_url": "http://judge/v1", "model": "j"}}
        )
        assert [e.label for e in endpoints] == ["judge"]  # the served model isn't an endpoint
        assert self._checks(
            {"eval_mode": "classification", "judge": {"base_url": "x", "model": "j"}}
        ) == ([], [])

    def test_malformed_url_is_config_error(self, monkeypatch) -> None:
        monkeypatch.setattr(eval_builder, "resolve_provider", lambda _n: None)
        _, errors = self._checks(
            {
                "model_name_or_path": "m",
                "judge": {"base_url": "http://judge:invalid/v1", "model": "j"},
            }
        )
        assert [(e.label, e.reason) for e in errors] == [("judge", "invalid_base_url")]

    def test_unknown_provider_name_is_config_error(self, monkeypatch) -> None:
        monkeypatch.setattr(eval_builder, "resolve_provider", lambda _n: None)
        _, errors = self._checks(
            {"model_name_or_path": "m", "judge": {"base_url": "anthropic", "model": "claude"}}
        )
        assert [(e.label, e.reason) for e in errors] == [("judge", "invalid_base_url")]


OPENAI_PROVIDER = {
    "name": "openai",
    "endpoint": "https://api.openai.com/v1",
    "provider_type": "openai",
    "api_key": "OPENAI_API_KEY",
}


class TestSdgHook:
    def _config(self, *aliases_used: str) -> dict[str, Any]:
        return {
            "model_configs": [
                {"alias": "teacher", "model": "gpt-5", "provider": "openai"},
                {"alias": "unused", "model": "dead-model", "provider": "nowhere"},
            ],
            "columns": [{"name": f"c{i}", "model_alias": a} for i, a in enumerate(aliases_used)],
        }

    def test_only_referenced_models_are_checked(self, monkeypatch) -> None:
        monkeypatch.setattr(
            model_endpoints,
            "resolve_provider",
            lambda n: OPENAI_PROVIDER if n == "openai" else None,
        )
        monkeypatch.setattr(
            sdg_builder,
            "inject_enabled_provider_keys",
            lambda env: env.update({"OPENAI_API_KEY": "sk-p"}),
        )
        endpoints, errors = sdg_builder.model_checks(self._config("teacher"), {})
        assert errors == []  # the unused model's unknown provider doesn't block
        assert [(e.label, e.model, e.api_key) for e in endpoints] == [
            ("teacher teacher", "gpt-5", "sk-p")
        ]

    def test_referenced_model_with_unknown_provider_is_config_error(self, monkeypatch) -> None:
        monkeypatch.setattr(model_endpoints, "resolve_provider", lambda _n: None)
        _, errors = sdg_builder.model_checks(self._config("unused"), {})
        assert len(errors) == 1 and isinstance(errors[0], ConfigError)
        assert errors[0].reason == "provider_not_configured"

    def test_embedding_and_image_columns_are_not_chat_probed(self, monkeypatch) -> None:
        # Their models answer other APIs; a chat probe would wrongly fail a valid job.
        monkeypatch.setattr(
            model_endpoints,
            "resolve_provider",
            lambda n: OPENAI_PROVIDER if n == "openai" else None,
        )
        monkeypatch.setattr(
            sdg_builder,
            "inject_enabled_provider_keys",
            lambda env: env.update({"OPENAI_API_KEY": "sk-p"}),
        )
        config = {
            "model_configs": [
                {"alias": "teacher", "model": "gpt-5", "provider": "openai"},
                {"alias": "embedder", "model": "text-embedding-3", "provider": "openai"},
                {"alias": "painter", "model": "gpt-image-1", "provider": "openai"},
            ],
            "columns": [
                {"name": "q", "column_type": "llm-text", "model_alias": "teacher"},
                {"name": "v", "column_type": "embedding", "model_alias": "embedder"},
                {"name": "img", "column_type": "image", "model_alias": "painter"},
            ],
        }
        endpoints, errors = sdg_builder.model_checks(config, {})
        assert errors == []
        assert [e.model for e in endpoints] == ["gpt-5"]

    def test_legacy_alias_field_counts_as_referenced(self, monkeypatch) -> None:
        monkeypatch.setattr(
            model_endpoints,
            "resolve_provider",
            lambda n: OPENAI_PROVIDER if n == "openai" else None,
        )
        monkeypatch.setattr(
            sdg_builder,
            "inject_enabled_provider_keys",
            lambda env: env.update({"OPENAI_API_KEY": "sk-p"}),
        )
        config = self._config()
        config["columns"] = [{"name": "c", "model_config_alias": "teacher"}]
        endpoints, _ = sdg_builder.model_checks(config, {})
        assert len(endpoints) == 1


class TestSdgGatewayTeacher:
    """data-designer's built-in gateway provider (what list_models reports on gateway-only
    clusters) isn't in the server's catalog; it resolves through gateway_url."""

    GATEWAY = "http://mlflow.amortized.svc.cluster.local:5000/gateway/mlflow/v1"

    def _config(self, model: str = "gpt-oss") -> dict[str, Any]:
        return {
            "model_configs": [{"alias": "text", "model": model, "provider": "gateway"}],
            "columns": [{"name": "q", "model_alias": "text"}],
        }

    @pytest.fixture(autouse=True)
    def _gateway_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_endpoints, "resolve_provider", lambda _n: None)
        monkeypatch.setattr(sdg_builder, "inject_enabled_provider_keys", lambda _env: None)
        monkeypatch.setattr(config_mod.settings, "gateway_url", self.GATEWAY)

    def test_gateway_teacher_is_checked_against_the_gateway_url(self) -> None:
        endpoints, errors = sdg_builder.model_checks(self._config(), {})
        assert errors == []
        [ep] = endpoints
        assert (ep.label, ep.base_url, ep.model, ep.provider_type) == (
            "teacher text",
            self.GATEWAY,
            "gpt-oss",
            "openai",
        )
        assert ep.api_key == "not-needed"
        [ep2], _ = sdg_builder.model_checks(self._config(), {"DD_API_KEY": "forwarded"})
        assert ep2.api_key == "not-needed"

    @pytest.mark.asyncio
    async def test_bad_gateway_model_is_probed_and_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(config_mod.settings, "model_check_url", "")
        monkeypatch.setattr(checker, "BACKOFF_SECONDS", (0.0, 0.0))
        seen: list[tuple[str, str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append((str(request.url), body["model"], request.headers["Authorization"]))
            return (
                httpx.Response(404)
                if body["model"] == "nope"
                else httpx.Response(
                    200, json={"choices": [{"message": {"role": "assistant", "content": "pong"}}]}
                )
            )

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            checker.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
        )
        report = await client.check_job_models("sdg", self._config("nope"))
        assert report.config_errors == []
        assert [(r.outcome, r.reason) for r in report.results] == [("failed", "not_found")]
        assert seen == [(f"{self.GATEWAY}/chat/completions", "nope", "Bearer not-needed")]

    def test_no_gateway_url_means_nothing_to_check(self, monkeypatch) -> None:
        monkeypatch.setattr(config_mod.settings, "gateway_url", "")
        assert sdg_builder.model_checks(self._config(), {}) == ([], [])

    def test_a_catalog_provider_named_gateway_wins(self, monkeypatch) -> None:
        custom = {**OPENAI_PROVIDER, "name": "gateway", "endpoint": "https://gw.example/v1"}
        monkeypatch.setattr(model_endpoints, "resolve_provider", lambda _n: custom)
        monkeypatch.setattr(
            sdg_builder,
            "inject_enabled_provider_keys",
            lambda env: env.update({"OPENAI_API_KEY": "sk-p"}),
        )
        [ep], _ = sdg_builder.model_checks(self._config(), {})
        assert (ep.base_url, ep.api_key) == ("https://gw.example/v1", "sk-p")
