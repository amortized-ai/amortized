"""FastAPI application entry point."""

import asyncio
import contextlib
import hmac
import logging
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from amortized.api import costs, documents, jobs, recipes
from amortized.backends.local import LocalBackend
from amortized.config import settings as _settings
from amortized.core.compute import get_all_backends, register_backend
from amortized.db import init_db
from amortized.mcp.server import create_mcp_server
from amortized.models import ConfigResponse, HealthResponse
from amortized.worker import cleanup_orphaned_jobs, worker_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("amortized")


def _load_backends() -> None:
    register_backend(LocalBackend())

    if _settings.compute_backend == "kubernetes":
        from amortized.backends.kubernetes import KubernetesBackend

        backend = KubernetesBackend(
            name="kubernetes",
            namespace=_settings.compute_namespace,
            image_registry=_settings.image_registry,
            image_pull_policy=_settings.image_pull_policy,
        )
        register_backend(backend)
        logger.info("Registered Kubernetes backend (namespace=%s)", _settings.compute_namespace)

    config_path = Path.home() / ".amortized" / "config.yaml"
    if not config_path.exists():
        return

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — skipping config.yaml backend loading")
        return

    try:
        config = yaml.safe_load(config_path.read_text())
    except Exception:
        logger.exception("Failed to read %s", config_path)
        return

    if not isinstance(config, dict):
        return

    default = config.get("compute", {}).get("default_backend", "")
    if default:
        _settings.default_backend = default

    forward_env = config.get("forward_env", [])
    if forward_env:
        _settings.forward_env = forward_env
        logger.info("Forwarding %d env vars to job containers", len(forward_env))

    gateway_url = config.get("gateway_url", "")
    if gateway_url:
        _settings.gateway_url = gateway_url
        logger.info("AI Gateway URL: %s", gateway_url)

    docling_url = config.get("docling_url", "")
    if docling_url:
        _settings.docling_url = docling_url
        logger.info("Docling-serve URL: %s", docling_url)

    backends = config.get("compute", {}).get("backends", {})
    for name, spec in backends.items():
        if not isinstance(spec, dict):
            continue
        backend_type = spec.get("type", "")
        if backend_type == "ssh":
            from amortized.backends.ssh import SSHBackend

            backend = SSHBackend(  # type: ignore[assignment]
                host=spec["host"],
                user=spec.get("user"),
                key_path=spec.get("key_path"),
                remote_base_dir=spec.get("remote_base_dir", "~/amortized-jobs"),
                name=name,
                container_runtime=spec.get("container_runtime", "podman"),
            )
            register_backend(backend)
            logger.info("Registered SSH backend %r (host=%s)", name, spec["host"])


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    _load_backends()
    await cleanup_orphaned_jobs()
    logger.info("Amortized runtime started")

    worker_task = asyncio.create_task(worker_loop())

    yield

    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task
    logger.info("Amortized runtime shutting down")


app = FastAPI(
    title="Amortized",
    description="Control plane for building task models",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_AUTH_SKIP_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def api_key_auth(request: Request, call_next):  # type: ignore[no-untyped-def]
    if not _settings.api_key:
        return await call_next(request)
    if request.url.path in _AUTH_SKIP_PATHS:
        return await call_next(request)
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={
                "code": "unauthorized",
                "message": "Missing or invalid Authorization header",
                "details": [],
            },
        )
    token = auth[len("Bearer ") :]
    if not hmac.compare_digest(token, _settings.api_key):
        return JSONResponse(
            status_code=401,
            content={"code": "unauthorized", "message": "Invalid API key", "details": []},
        )
    return await call_next(request)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": f"http_{exc.status_code}",
            "message": detail,
            "details": [],
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "code": "validation_error",
            "message": "Request validation failed",
            "details": [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ],
        },
    )


app.include_router(jobs.router)
app.include_router(recipes.router)
app.include_router(recipes.recipe_jobs_router)
app.include_router(costs.router)
app.include_router(documents.router)

from amortized.api.models import router as models_router  # noqa: E402

app.include_router(models_router)

create_mcp_server(app)


def _detect_gpu() -> dict[str, object]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "count": torch.cuda.device_count(),
                "devices": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ],
            }
    except ImportError:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    return {
        "available": nvidia_smi is not None,
        "count": 0,
        "devices": [],
        "note": "torch not installed; nvidia-smi " + ("found" if nvidia_smi else "not found"),
    }


@app.get("/api/v1/health", response_model=HealthResponse, operation_id="health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "gpu": _detect_gpu(),
    }


@app.get("/api/v1/config", response_model=ConfigResponse, operation_id="get_config")
async def get_config() -> ConfigResponse:
    mlflow_gateway_uri = _settings.gateway_url or ""
    return ConfigResponse(
        default_compute_backend=_settings.resolved_default_backend,
        compute_namespace=_settings.compute_namespace,
        mlflow_tracking_uri=_settings.mlflow_tracking_uri,
        mlflow_gateway_uri=mlflow_gateway_uri,
        docling_enabled=bool(_settings.docling_url),
        image_registry=_settings.image_registry,
        available_backends=list(get_all_backends().keys()),
    )
