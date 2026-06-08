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

from amortized.api import (
    agent_routes,
    artifacts,
    compute,
    datasets,
    estimate,
    evaluators,
    event_ingest,
    events,
    flows,
    job_types,
    jobs,
    judge,
    recipes,
    settings,
    ws,
)
from amortized.backends.local import LocalBackend
from amortized.config import settings as _settings
from amortized.core.compute import register_backend
from amortized.db import get_db, init_db
from amortized.mcp.server import create_mcp_server
from amortized.models import HealthResponse
from amortized.worker import _monitor_heartbeats, cleanup_orphaned_jobs, worker_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("amortized")


def _load_backends() -> None:
    """Register compute backends from config file and always include local."""
    register_backend(LocalBackend())

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
        from amortized.config import settings

        settings.default_backend = default

    backends = config.get("compute", {}).get("backends", {})
    for name, spec in backends.items():
        if not isinstance(spec, dict):
            continue
        backend_type = spec.get("type", "")
        if backend_type == "ssh":
            from amortized.backends.ssh import SSHBackend

            backend = SSHBackend(
                host=spec["host"],
                user=spec.get("user"),
                key_path=spec.get("key_path"),
                remote_base_dir=spec.get("remote_base_dir", "~/amortized-jobs"),
                name=name,
                container_runtime=spec.get("container_runtime", "docker"),
            )
            register_backend(backend)
            logger.info("Registered SSH backend %r (host=%s)", name, spec["host"])


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize database and start background worker on startup."""
    await init_db()
    async for db in get_db():
        await evaluators.seed_default_evaluators(db)
    await cleanup_orphaned_jobs()
    _load_backends()
    logger.info("Amortized runtime started")

    # Start background worker and heartbeat monitor
    worker_task = asyncio.create_task(worker_loop())
    heartbeat_task = asyncio.create_task(_monitor_heartbeats())

    yield

    # Shutdown worker and heartbeat monitor
    worker_task.cancel()
    heartbeat_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task
    with contextlib.suppress(asyncio.CancelledError):
        await heartbeat_task
    logger.info("Amortized runtime shutting down")


app = FastAPI(
    title="Amortized Runtime",
    description="AI model customization runtime API",
    version="0.1.0",
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
    if isinstance(exc.detail, list):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": f"http_{exc.status_code}",
                "message": "Request failed",
                "details": [{"msg": str(e)} for e in exc.detail],
            },
        )
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
app.include_router(job_types.router)
app.include_router(job_types.job_types_router)
app.include_router(events.router)
app.include_router(event_ingest.router)
app.include_router(artifacts.router)
app.include_router(flows.router)
app.include_router(estimate.router)
app.include_router(datasets.router)
app.include_router(ws.router)
app.include_router(recipes.router)
app.include_router(recipes.recipe_jobs_router)
app.include_router(compute.router)
app.include_router(judge.router)
app.include_router(agent_routes.router)
app.include_router(evaluators.router)
app.include_router(settings.router)

create_mcp_server(app)


def _detect_gpu() -> dict[str, object]:
    """Detect GPU availability."""
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

    # Fallback: check for nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    return {
        "available": nvidia_smi is not None,
        "count": 0,
        "devices": [],
        "note": "torch not installed; nvidia-smi " + ("found" if nvidia_smi else "not found"),
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> dict[str, object]:
    """Health check endpoint with GPU info."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "gpu": _detect_gpu(),
    }
