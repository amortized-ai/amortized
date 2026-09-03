"""Auto-generated MCP server from the FastAPI OpenAPI spec.

Uses fastapi-mcp to expose all API endpoints as MCP tools.
External AI agents connect via the MCP HTTP transport at /mcp.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from fastapi import Request, Response
from fastapi_mcp import FastApiMCP
from fastapi_mcp.transport.http import FastApiHttpSessionManager
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("amortized.mcp")


class _StatelessHttpSessionManager(FastApiHttpSessionManager):  # type: ignore[misc]
    """``FastApiHttpSessionManager`` forced into stateless mode.

    fastapi-mcp 0.4.0 hardcodes ``stateless=False`` when it builds the underlying
    ``StreamableHTTPSessionManager`` (in ``_ensure_session_manager_started``), so
    the MCP server hands each client an in-memory ``mcp-session-id``. Any
    amortized-server restart drops that state; opencode (Morty) keeps sending its
    now-unknown session id and every tool call fails with "session not recognized",
    and opencode does not auto-reconnect. In stateless mode no session id is issued
    or required, so tool calls keep working across server redeploys. Our MCP tools
    are stateless HTTP proxies back to this same FastAPI app, so there is no
    per-session server state to lose.
    """

    async def _ensure_session_manager_started(self) -> None:
        if self._manager_started:  # type: ignore[has-type]
            return
        async with self._startup_lock:
            if self._manager_started:  # type: ignore[has-type]
                return
            self._session_manager = StreamableHTTPSessionManager(
                app=self.mcp_server,
                event_store=self.event_store,
                json_response=self.json_response,
                stateless=True,
                security_settings=self.security_settings,
            )

            async def _run() -> None:
                async with self._session_manager.run():
                    await asyncio.Event().wait()

            self._manager_task = asyncio.create_task(_run())
            self._manager_started = True
            await asyncio.sleep(0.1)


def create_mcp_server(app: FastAPI) -> FastApiMCP:
    """Create and mount a stateless MCP server from the FastAPI app's OpenAPI spec.

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
            "get_turn_agent_session__session_id__turn__turn_id__get",
            "generate_title_agent_title_post",
            "agent_health_agent_health_get",
        ],
    )
    _mount_http_stateless(mcp, app)
    return mcp


def _mount_http_stateless(mcp: FastApiMCP, app: FastAPI, mount_path: str = "/mcp") -> None:
    """Register the MCP HTTP route using a stateless session manager.

    Mirrors ``FastApiMCP.mount_http`` (same route methods and ``operation_id`` so the
    endpoint is excluded from the OpenAPI/tool set identically) but swaps in
    :class:`_StatelessHttpSessionManager`. We set no ``auth_config``, so the library's
    ``_setup_auth`` (a no-op in that case) is intentionally not called.
    """
    transport = _StatelessHttpSessionManager(mcp_server=mcp.server)

    @app.api_route(
        mount_path,
        methods=["GET", "POST", "DELETE"],
        include_in_schema=False,
        operation_id="mcp_http",
    )
    async def handle_mcp_streamable_http(request: Request) -> Response:
        response: Response = await transport.handle_fastapi_request(request)
        return response

    mcp._http_transport = transport
    app.state.mcp_transport = transport
    logger.info("MCP HTTP server (stateless) listening at %s", mount_path)


async def shutdown_mcp_transport(transport: Any) -> None:
    """Cancel the stateless MCP transport's background manager task on app shutdown.

    ``_ensure_session_manager_started`` launches a task that blocks forever inside
    ``StreamableHTTPSessionManager.run()``; nothing else cancels it. Cancelling exits
    that context manager cleanly. No-op if the transport never started a session
    manager (no MCP request was served).
    """
    if transport is None:
        return
    task = getattr(transport, "_manager_task", None)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
