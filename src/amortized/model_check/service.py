"""HTTP service for the model-liveness checker.

Opt-in: by default the server runs the same checker in-process. Run this standalone where
independent scaling or controlled egress justifies it:
``python3 -m amortized.model_check serve --port 7351 --ssl-certfile … --ssl-keyfile …``.
Callers send each endpoint's resolved key in the request, so serve TLS (the server only talks to
an https service URL); the service stores nothing and never logs or echoes keys. Requests are
bounded (endpoints per request, deadline, in-flight probes) so a burst can't exhaust it into
false results.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, SecretStr

from amortized.model_check import checker
from amortized.model_check.types import ModelEndpoint

logger = logging.getLogger("amortized.model_check")

MAX_ENDPOINTS_PER_REQUEST = 8
MAX_DEADLINE_SECONDS = 30.0


class EndpointIn(BaseModel):
    label: str = Field(..., max_length=200)
    base_url: str = Field(..., max_length=2048)
    model: str = Field(..., max_length=500)
    provider_type: Literal["openai", "anthropic"] = "openai"
    api_key: SecretStr = Field(SecretStr(""), max_length=8192)


class CheckRequest(BaseModel):
    deadline_seconds: float = Field(20.0, gt=0)
    endpoints: list[EndpointIn]
    # The server's approved prefixes for private services; the policy is applied here too.
    approved_urls: list[str] = Field(default_factory=list, max_length=200)


app = FastAPI(title="amortized model check", docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default echoes the offending input, which would include API keys.
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/check", response_model=None)
async def check(request: CheckRequest) -> dict[str, Any] | JSONResponse:
    count = len(request.endpoints)
    if count > MAX_ENDPOINTS_PER_REQUEST:
        return JSONResponse(
            status_code=413,
            content={"detail": f"at most {MAX_ENDPOINTS_PER_REQUEST} endpoints per request"},
        )
    # Reject instead of queueing: a queued probe would burn the caller's deadline and come back
    # as a false failure. The caller reports a 503 as unverified/checker_busy.
    if not checker.try_reserve(count):
        return JSONResponse(status_code=503, content={"detail": "model check service is busy"})
    try:
        endpoints = [
            ModelEndpoint(
                label=e.label,
                base_url=e.base_url,
                model=e.model,
                provider_type=e.provider_type,
                api_key=e.api_key.get_secret_value(),
            )
            for e in request.endpoints
        ]
        results = await checker.check_endpoints(
            endpoints,
            min(request.deadline_seconds, MAX_DEADLINE_SECONDS),
            tuple(request.approved_urls),
        )
    finally:
        checker.release(count)
    for r in results:
        logger.info(
            "model check %s %s at %s: %s/%s", r.label, r.model, r.base_url, r.outcome, r.reason
        )
    return {"results": [r.to_dict() for r in results]}


def main() -> None:
    parser = argparse.ArgumentParser(description="amortized model-liveness check service")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the HTTP check service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=7351)
    serve.add_argument("--limit-concurrency", type=int, default=64)
    serve.add_argument("--ssl-certfile", default="", help="TLS certificate (PEM)")
    serve.add_argument("--ssl-keyfile", default="", help="TLS private key (PEM)")
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Request lines from httpx are redacted by the checker; httpcore's connection-level logs add
    # nothing here, so keep them off even if the level is raised.
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        limit_concurrency=args.limit_concurrency,
        ssl_certfile=args.ssl_certfile or None,
        ssl_keyfile=args.ssl_keyfile or None,
    )
