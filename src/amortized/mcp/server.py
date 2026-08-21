"""Auto-generated MCP server from the FastAPI OpenAPI spec.

Uses fastapi-mcp to expose all API endpoints as MCP tools.
External AI agents connect via the MCP HTTP transport at /mcp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi_mcp import FastApiMCP

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_mcp_server(app: FastAPI) -> FastApiMCP:
    """Create and mount an MCP server from the FastAPI app's OpenAPI spec.

    MCP resources (system://capabilities, jobs://recent, recipes://{name})
    are deferred — fastapi-mcp only supports auto-generated tools from
    OpenAPI endpoints, not custom resource registration.
    """
    mcp = FastApiMCP(
        app,
        name="amortized",
        description=(
            "Amortized — control plane for building task models."
            " Submit SDG and training jobs. Browse recipes. Track job lifecycle."
        ),
        describe_all_responses=True,
        describe_full_response_schema=True,
        exclude_operations=[
            "create_sdg_job",
            "create_training_job",
            "submit_recipe_job",
            "create_session_agent_session_post",
            "send_message_agent_session__session_id__message_post",
            "get_session_messages_agent_session__session_id__message_get",
            "get_pending_agent_session__session_id__pending_get",
            "generate_title_agent_title_post",
            "agent_health_agent_health_get",
        ],
    )
    mcp.mount_http(app)
    return mcp
