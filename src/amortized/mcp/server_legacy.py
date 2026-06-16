"""Legacy MCP server using fastapi-mcp auto-generation.

Kept for backward compatibility behind AMORTIZED_LEGACY_MCP=1 env flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi_mcp import FastApiMCP

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_mcp_server(app: FastAPI) -> FastApiMCP:
    """Create and mount an MCP server from the FastAPI app's OpenAPI spec."""
    mcp = FastApiMCP(
        app,
        name="amortized",
        description=(
            "Amortized — AI model customization runtime."
            " Submit training/SDG jobs, track progress, manage artifacts."
        ),
        describe_all_responses=True,
        describe_full_response_schema=True,
    )
    mcp.mount_http(app)
    return mcp
