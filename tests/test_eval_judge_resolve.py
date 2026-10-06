"""Unit tests for `_resolve_eval_judge` — the gateway judge auto-corrector.

Regression guard for the silent zero-score bug: a rubric judge set to the
gateway endpoint's underlying provider id (``openai/gpt-oss-120b``) 404s on
every call, so the eval "succeeds" with no scores. The resolver rewrites it to
the gateway endpoint name (``gpt-oss``) or errors on a truly unknown model.

DB-free: the resolver reads `amortized.config.settings` and builds an
`MLflowClient`, both monkeypatched here.
"""

import pytest

import amortized.config as config_mod
import amortized.core.mlflow_client as mlflow_mod
from amortized.api import jobs

_GATEWAY = "http://gw.svc:5000/gateway/mlflow/v1"


class _FakeGWClient:
    def __init__(self, *_a, **_k) -> None:
        pass

    async def list_gateway_models(self):
        return [
            {"name": "gpt-oss", "provider": "gateway", "model_name": "openai/gpt-oss-120b"},
            {"name": "gpt-4o-mini", "provider": "gateway", "model_name": "gpt-4o-mini"},
        ]


@pytest.fixture
def gateway(monkeypatch):
    """Configure a gateway and a fake client; yield a toggle to disable it."""
    monkeypatch.setattr(config_mod.settings, "mlflow_tracking_uri", "http://mlflow", raising=False)
    monkeypatch.setattr(config_mod.settings, "gateway_url", _GATEWAY, raising=False)
    monkeypatch.setattr(mlflow_mod, "MLflowClient", _FakeGWClient)
    return monkeypatch


@pytest.mark.asyncio
class TestResolveEvalJudge:
    async def test_rewrites_provider_id_to_gateway_name(self, gateway) -> None:
        cfg = {
            "rubric": [{"name": "acc", "description": "d"}],
            "judge": {"model": "openai/gpt-oss-120b", "base_url": _GATEWAY},
        }
        errors, warnings = await jobs._resolve_eval_judge(cfg)
        assert errors == []
        assert cfg["judge"]["model"] == "gpt-oss"  # corrected to the endpoint name
        assert cfg["judge"]["base_url"] == _GATEWAY
        assert warnings and "zero scores" in warnings[0]

    async def test_valid_gateway_name_passes_and_sets_base_url(self, gateway) -> None:
        cfg = {"judge": {"model": "gpt-oss"}}  # base_url missing
        errors, warnings = await jobs._resolve_eval_judge(cfg)
        assert errors == [] and warnings == []
        assert cfg["judge"]["model"] == "gpt-oss"
        assert cfg["judge"]["base_url"] == _GATEWAY  # pointed at the gateway

    async def test_unknown_model_is_a_hard_error(self, gateway) -> None:
        cfg = {"judge": {"model": "mystery-model", "base_url": _GATEWAY}}
        errors, warnings = await jobs._resolve_eval_judge(cfg)
        assert warnings == []
        assert errors and "not a known MLflow gateway endpoint" in errors[0]
        assert "gpt-oss" in errors[0]  # lists valid endpoint names
        assert cfg["judge"]["model"] == "mystery-model"  # unchanged

    async def test_noop_when_no_gateway_configured(self, monkeypatch) -> None:
        monkeypatch.setattr(config_mod.settings, "gateway_url", "", raising=False)
        cfg = {"judge": {"model": "openai/gpt-oss-120b"}}
        errors, warnings = await jobs._resolve_eval_judge(cfg)
        assert errors == [] and warnings == []
        assert cfg["judge"]["model"] == "openai/gpt-oss-120b"  # left untouched

    async def test_leaves_direct_keyed_endpoint_alone(self, gateway) -> None:
        cfg = {
            "judge": {
                "model": "gpt-4o",  # not a gateway endpoint, but direct + keyed
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-xxx",
            }
        }
        errors, warnings = await jobs._resolve_eval_judge(cfg)
        assert errors == [] and warnings == []
        assert cfg["judge"]["model"] == "gpt-4o"

    async def test_noop_without_judge(self, gateway) -> None:
        cfg = {"rubric": [{"name": "acc", "description": "d"}]}
        assert await jobs._resolve_eval_judge(cfg) == ([], [])
