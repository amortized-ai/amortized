"""Tests for MLflowClient gateway-model mapping and the gateway MLflow choice."""

from __future__ import annotations

import pytest

from amortized.core.mlflow_client import MLflowClient, gateway_mlflow_client


@pytest.mark.asyncio
async def test_list_gateway_models_reports_gateway_provider(monkeypatch) -> None:
    """Gateway-served models must report provider 'gateway' (the provider a job
    uses to reach them), not the endpoint's underlying type.

    The SDG skill copies the provider verbatim (<from-list_models>). On a
    gateway-only cluster the underlying type (e.g. 'openai') is unresolvable, so
    reporting it makes SDG fail with "No provider named 'openai' registered". The
    underlying model stays in model_name.
    """
    client = MLflowClient("http://mlflow:5000")

    async def fake_endpoints() -> list[dict]:
        return [
            {
                "name": "gpt-oss",
                "model_mappings": [
                    {
                        "model_definition": {
                            "provider": "openai",
                            "model_name": "openai/gpt-oss-120b",
                        }
                    }
                ],
            }
        ]

    monkeypatch.setattr(client, "list_gateway_endpoints", fake_endpoints)

    models = await client.list_gateway_models()

    assert models == [
        {"name": "gpt-oss", "provider": "gateway", "model_name": "openai/gpt-oss-120b"}
    ]


class TestGatewayMlflowClient:
    """The model list comes from the MLflow behind gateway_url, the gateway jobs
    call, so it can't name endpoints that gateway lacks."""

    @pytest.fixture
    def settings(self, monkeypatch, tmp_path):
        import amortized.config as config_mod

        token = tmp_path / "token"
        token.write_text("sa-token")
        s = config_mod.settings
        monkeypatch.setattr(s, "mlflow_tracking_token_file", str(token))
        monkeypatch.setattr(s, "mlflow_workspace", "amortized-user-jobs")
        monkeypatch.setattr(s, "mlflow_ca_bundle", "")
        monkeypatch.setattr(s, "mlflow_tracking_insecure_tls", False)

        def configure(tracking: str, gateway: str):
            monkeypatch.setattr(s, "mlflow_tracking_uri", tracking)
            monkeypatch.setattr(s, "gateway_url", gateway)

        return configure

    def test_separate_gateway_mlflow_gets_no_token_or_workspace(self, settings) -> None:
        # KIND: private workspaced tracking MLflow, shared gateway MLflow (workspaces off,
        # rejects the header).
        settings(
            "https://mlflow-ws.amortized-user.svc:5000",
            "http://mlflow.amortized.svc:5000/gateway/mlflow/v1",
        )
        client = gateway_mlflow_client()
        assert client is not None
        assert client._base == "http://mlflow.amortized.svc:5000"
        assert client._auth.headers() == {}

    def test_gateway_on_the_tracking_mlflow_keeps_auth(self, settings) -> None:
        settings("https://mlflow.ns.svc:5000", "https://mlflow.ns.svc:5000/gateway/mlflow/v1")
        client = gateway_mlflow_client()
        assert client is not None
        assert client._base == "https://mlflow.ns.svc:5000"
        assert client._auth.headers() == {
            "Authorization": "Bearer sa-token",
            "X-MLFLOW-WORKSPACE": "amortized-user-jobs",
        }

    @pytest.mark.parametrize("gateway", ["", "http://litellm:4000/v1"])
    def test_without_an_mlflow_gateway_url_uses_the_tracking_mlflow(
        self, settings, gateway: str
    ) -> None:
        # RHOAI has no gateway_url; a non-MLflow gateway URL names no MLflow either.
        settings("https://mlflow.ns.svc:5000", gateway)
        client = gateway_mlflow_client()
        assert client is not None
        assert client._base == "https://mlflow.ns.svc:5000"
        assert "Authorization" in client._auth.headers()

    def test_no_mlflow_at_all(self, settings) -> None:
        settings("", "")
        assert gateway_mlflow_client() is None


@pytest.mark.asyncio
async def test_list_models_reads_the_gateway_mlflow(monkeypatch) -> None:
    import amortized.api.models as models_api
    import amortized.config as config_mod

    class FakeGatewayClient:
        async def list_gateway_models(self) -> list[dict]:
            return [{"name": "gpt-oss", "provider": "gateway", "model_name": "openai/gpt-oss"}]

    gateway = "http://mlflow.amortized.svc:5000/gateway/mlflow/v1"
    monkeypatch.setattr(config_mod.settings, "gateway_url", gateway)
    monkeypatch.setattr(models_api, "gateway_mlflow_client", lambda: FakeGatewayClient())

    response = await models_api.list_models()

    assert [m.name for m in response.models] == ["gpt-oss"]
    assert response.gateway_url == gateway
