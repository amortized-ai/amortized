"""MCP server using fastmcp with httpx ASGI transport.

Tools call back into the FastAPI app in-process via httpx.ASGITransport,
avoiding real network round-trips.  HTTP 4xx/5xx responses are translated
into structured MCP error messages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from fastmcp import FastMCP

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "amortized",
    instructions=(
        "Amortized - AI model customization runtime."
        " Submit training/SDG jobs, track progress, manage artifacts."
    ),
)

_client: httpx.AsyncClient | None = None


async def _call(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> Any:
    """Call a FastAPI endpoint in-process via ASGI transport.

    Raises ``ValueError`` with a structured message on HTTP 4xx/5xx so
    that fastmcp surfaces the error to the caller.
    """
    if _client is None:
        raise RuntimeError("MCP ASGI client not initialised; call init_mcp_client first")

    response = await _client.request(method, path, params=params, json=json)

    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:
            body = {"message": response.text or "Unknown error"}
        code = body.get("code", f"http_{response.status_code}")
        message = body.get("message", response.reason_phrase)
        details = body.get("details", [])
        raise ValueError(
            f"[{code}] {message}"
            + (f" | details: {details}" if details else "")
        )

    if response.status_code == 204:
        return None
    return response.json()


def init_mcp_client(app: FastAPI) -> None:
    """Bind the httpx ASGI client to the given FastAPI app."""
    global _client
    _client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://amortized",
    )
    logger.info("MCP ASGI transport client initialised")
