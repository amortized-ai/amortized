"""Tests for the Typer CLI."""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from amortized.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _mock_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMORTIZED_API_URL", "http://test:8000")


def _mock_response(status_code: int = 200, json_data: object = None) -> object:
    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = status_code
            self._json = json_data

        def json(self) -> object:
            return self._json

        @property
        def text(self) -> str:
            return json.dumps(self._json) if self._json else ""

    return FakeResponse()


class _FakeClient:
    def __init__(self, responses: dict[str, object] | None = None) -> None:
        self._responses = responses or {}
        self._default_resp = _mock_response(200, {"status": "ok"})

    def get(self, url: str, **kwargs: object) -> object:
        return self._responses.get(("GET", url), self._default_resp)

    def post(self, url: str, **kwargs: object) -> object:
        return self._responses.get(("POST", url), self._default_resp)

    def delete(self, url: str, **kwargs: object) -> object:
        return self._responses.get(("DELETE", url), self._default_resp)

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: object) -> None:
        pass


class TestHealth:
    def test_health_ok(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/health"): _mock_response(
                    200,
                    {
                        "status": "ok",
                        "gpu": {"available": False},
                    },
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_health_connection_error(self) -> None:
        import httpx

        def _raise_connect(*a: object, **kw: object) -> object:
            raise httpx.ConnectError("refused")

        with patch("amortized.cli.main._client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = lambda s, *a: None
            mock_client.return_value.get = _raise_connect
            result = runner.invoke(app, ["health"])
        assert result.exit_code == 1


class TestTypes:
    def test_list_types(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/job-types"): _mock_response(
                    200,
                    [
                        {"name": "training", "description": "LoRA SFT training"},
                        {"name": "sdg", "description": "Synthetic data generation"},
                    ],
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["types"])
        assert result.exit_code == 0
        assert "training" in result.output
        assert "sdg" in result.output


class TestJobs:
    def test_list_jobs_empty(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/jobs"): _mock_response(200, []),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["jobs"])
        assert result.exit_code == 0
        assert "Jobs" in result.output

    def test_list_jobs_with_data(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/jobs"): _mock_response(
                    200,
                    [
                        {
                            "id": "job_123",
                            "type": "training",
                            "status": "running",
                            "created_at": "2026-01-01T00:00:00Z",
                        },
                    ],
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["jobs"])
        assert result.exit_code == 0
        assert "job_123" in result.output
        assert "training" in result.output


class TestJobDetail:
    def test_get_job(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/jobs/job_abc"): _mock_response(
                    200,
                    {
                        "id": "job_abc",
                        "type": "training",
                        "status": "completed",
                        "created_at": "2026-01-01T00:00:00Z",
                        "config": {"model_path": "test"},
                    },
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["job", "job_abc"])
        assert result.exit_code == 0
        assert "job_abc" in result.output

    def test_get_job_not_found(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/jobs/nope"): _mock_response(404, {"detail": "not found"}),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["job", "nope"])
        assert result.exit_code == 1


class TestCancel:
    def test_cancel_job(self) -> None:
        fake = _FakeClient(
            {
                ("DELETE", "/api/v1/jobs/job_xyz"): _mock_response(
                    200,
                    {
                        "id": "job_xyz",
                        "type": "training",
                        "status": "cancelled",
                        "created_at": "2026-01-01T00:00:00Z",
                    },
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["cancel", "job_xyz"])
        assert result.exit_code == 0
        assert "Cancelled" in result.output


class TestRecipes:
    def test_list_recipes(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/recipes"): _mock_response(
                    200,
                    [
                        {
                            "name": "llama3/8b-lora-sft",
                            "type": "training",
                            "description": "LoRA SFT",
                        },
                    ],
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["recipes"])
        assert result.exit_code == 0
        assert "llama3/8b-lora-sft" in result.output

    def test_show_recipe(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/recipes/llama3/8b-lora-sft"): _mock_response(
                    200,
                    {
                        "name": "llama3/8b-lora-sft",
                        "type": "training",
                        "config": {"model_path": "meta-llama/Llama-3-8B"},
                    },
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["recipe", "llama3/8b-lora-sft"])
        assert result.exit_code == 0
        assert "llama3/8b-lora-sft" in result.output


class TestSubmit:
    def test_submit_with_config(self) -> None:
        fake = _FakeClient(
            {
                ("POST", "/api/v1/jobs"): _mock_response(
                    201,
                    {
                        "id": "job_new",
                        "type": "training",
                        "status": "pending",
                        "created_at": "2026-01-01T00:00:00Z",
                        "config": {"model_path": "test"},
                    },
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(
                app,
                ["submit", "training", "--config", '{"model_path": "test"}'],
            )
        assert result.exit_code == 0
        assert "job_new" in result.output

    def test_submit_with_recipe(self) -> None:
        fake = _FakeClient(
            {
                ("POST", "/api/v1/jobs/recipe"): _mock_response(
                    201,
                    {
                        "id": "job_recipe",
                        "type": "training",
                        "status": "pending",
                        "created_at": "2026-01-01T00:00:00Z",
                    },
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(
                app,
                ["submit", "training", "--recipe", "llama3/8b-lora-sft"],
            )
        assert result.exit_code == 0
        assert "job_recipe" in result.output


class TestBackends:
    def test_list_backends(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/compute"): _mock_response(
                    200,
                    [
                        {"name": "local", "capabilities": ["LOG_STREAM", "STOP"]},
                    ],
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["backends"])
        assert result.exit_code == 0
        assert "local" in result.output


class TestUpload:
    def test_upload_file(self, tmp_path: object) -> None:
        test_file = tmp_path / "data.jsonl"  # type: ignore[operator]
        test_file.write_text('{"text": "hello"}\n')

        fake = _FakeClient(
            {
                ("POST", "/api/v1/artifacts"): _mock_response(
                    201,
                    {
                        "id": "art_uploaded",
                        "name": "data.jsonl",
                        "artifact_type": "dataset",
                        "location": str(test_file),
                    },
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["upload", str(test_file)])
        assert result.exit_code == 0
        assert "art_uploaded" in result.output

    def test_upload_with_name_and_type(self, tmp_path: object) -> None:
        test_file = tmp_path / "model.safetensors"  # type: ignore[operator]
        test_file.write_text("fake model data")

        fake = _FakeClient(
            {
                ("POST", "/api/v1/artifacts"): _mock_response(
                    201,
                    {
                        "id": "art_model",
                        "name": "my-model",
                        "artifact_type": "adapter_weights",
                        "location": str(test_file),
                    },
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(
                app, ["upload", str(test_file), "--type", "adapter_weights", "--name", "my-model"]
            )
        assert result.exit_code == 0
        assert "my-model" in result.output

    def test_upload_nonexistent_file(self) -> None:
        result = runner.invoke(app, ["upload", "/nonexistent/file.jsonl"])
        assert result.exit_code == 1


class TestArtifacts:
    def test_list_artifacts(self) -> None:
        fake = _FakeClient(
            {
                ("GET", "/api/v1/artifacts"): _mock_response(
                    200,
                    [
                        {
                            "id": "art_1",
                            "name": "model-v1",
                            "artifact_type": "model",
                            "job_id": "job_1",
                        },
                    ],
                ),
            }
        )
        with patch("amortized.cli.main._client", return_value=fake):
            result = runner.invoke(app, ["artifacts"])
        assert result.exit_code == 0
        assert "model-v1" in result.output
