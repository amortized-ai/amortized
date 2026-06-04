"""Typer CLI — thin REST client wrapping the Amortized API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="amortized",
    help="Amortized — AI Model Customization Studio CLI",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _discover_url() -> str:
    env = os.environ.get("AMORTIZED_API_URL")
    if env:
        return env.rstrip("/")

    config_path = Path.home() / ".amortized" / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            data = yaml.safe_load(config_path.read_text())
            if isinstance(data, dict) and data.get("api_url"):
                return str(data["api_url"]).rstrip("/")
        except Exception:
            pass

    return "http://localhost:8000"


def _client() -> httpx.Client:
    return httpx.Client(base_url=_discover_url(), timeout=30.0)


def _handle_response(resp: httpx.Response) -> Any:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        err_console.print(f"[red]Error ({resp.status_code}):[/red] {detail}")
        raise typer.Exit(1)
    return resp.json()


# ---------------------------------------------------------------------------
# amortized up
# ---------------------------------------------------------------------------
@app.command()
def up(
    host: Annotated[str, typer.Option(help="Bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
) -> None:
    """Start the Amortized API server."""
    console.print(f"[bold green]Starting Amortized server[/bold green] on {host}:{port}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "amortized.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        check=False,
    )


# ---------------------------------------------------------------------------
# amortized submit
# ---------------------------------------------------------------------------
@app.command()
def submit(
    job_type: Annotated[str, typer.Argument(help="Job type (e.g. training, sdg)")],
    config: Annotated[str | None, typer.Option("--config", help="JSON config string")] = None,
    recipe: Annotated[str | None, typer.Option("--recipe", help="Recipe name")] = None,
    set_values: Annotated[
        list[str] | None, typer.Option("--set", help="Override KEY=VALUE")
    ] = None,
) -> None:
    """Submit a job by type or recipe."""
    with _client() as client:
        if recipe:
            overrides: dict[str, Any] = {}
            for kv in set_values or []:
                if "=" not in kv:
                    err_console.print(f"[red]Invalid --set format:[/red] {kv} (expected KEY=VALUE)")
                    raise typer.Exit(1)
                k, v = kv.split("=", 1)
                try:
                    overrides[k] = json.loads(v)
                except json.JSONDecodeError:
                    overrides[k] = v
            body: dict[str, Any] = {"recipe": recipe, "overrides": overrides}
            resp = client.post("/api/v1/jobs/recipe", json=body)
        else:
            cfg: dict[str, Any] = {}
            if config:
                try:
                    cfg = json.loads(config)
                except json.JSONDecodeError as exc:
                    err_console.print(f"[red]Invalid JSON config:[/red] {exc}")
                    raise typer.Exit(1) from exc
            for kv in set_values or []:
                if "=" not in kv:
                    err_console.print(f"[red]Invalid --set format:[/red] {kv} (expected KEY=VALUE)")
                    raise typer.Exit(1)
                k, v = kv.split("=", 1)
                try:
                    cfg[k] = json.loads(v)
                except json.JSONDecodeError:
                    cfg[k] = v
            body = {"type": job_type, "config": cfg}
            resp = client.post("/api/v1/jobs", json=body)

        data = _handle_response(resp)
        _print_job_panel(data)


# ---------------------------------------------------------------------------
# amortized jobs
# ---------------------------------------------------------------------------
@app.command()
def jobs(
    status: Annotated[str | None, typer.Option("--status", help="Filter by status")] = None,
    job_type: Annotated[str | None, typer.Option("--type", help="Filter by type")] = None,
) -> None:
    """List jobs."""
    with _client() as client:
        params: dict[str, str] = {}
        if status:
            params["status"] = status
        if job_type:
            params["type"] = job_type
        resp = client.get("/api/v1/jobs", params=params)
        data = _handle_response(resp)

    table = Table(title="Jobs")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Status")
    table.add_column("Created")

    for j in data:
        status_str = _colorize_status(j.get("status", ""))
        table.add_row(
            j.get("id", ""),
            j.get("type", ""),
            status_str,
            j.get("created_at", ""),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# amortized job <id>
# ---------------------------------------------------------------------------
@app.command()
def job(
    job_id: Annotated[str, typer.Argument(help="Job ID")],
) -> None:
    """Get job details."""
    with _client() as client:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        data = _handle_response(resp)
    _print_job_panel(data)


# ---------------------------------------------------------------------------
# amortized logs <id>
# ---------------------------------------------------------------------------
@app.command()
def logs(
    job_id: Annotated[str, typer.Argument(help="Job ID")],
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Follow live events")] = False,
) -> None:
    """Stream job logs/events."""
    with _client() as client:
        if not follow:
            resp = client.get(f"/api/v1/jobs/{job_id}/events")
            events = _handle_response(resp)
            for event in events:
                _print_event(event)
            return

        since: str | None = None
        while True:
            params: dict[str, str] = {}
            if since:
                params["since"] = since
            resp = client.get(f"/api/v1/jobs/{job_id}/events", params=params)
            if resp.status_code >= 400:
                _handle_response(resp)
                return
            events = resp.json()
            for event in events:
                _print_event(event)
                since = event.get("timestamp")

            job_resp = client.get(f"/api/v1/jobs/{job_id}")
            if job_resp.status_code == 200:
                job_data = job_resp.json()
                if job_data.get("status") in ("succeeded", "completed", "failed", "cancelled"):
                    break
            time.sleep(2)


# ---------------------------------------------------------------------------
# amortized cancel <id>
# ---------------------------------------------------------------------------
@app.command()
def cancel(
    job_id: Annotated[str, typer.Argument(help="Job ID")],
) -> None:
    """Cancel a job."""
    with _client() as client:
        resp = client.delete(f"/api/v1/jobs/{job_id}")
        data = _handle_response(resp)
    console.print(f"[yellow]Cancelled[/yellow] job [cyan]{data.get('id', job_id)}[/cyan]")


# ---------------------------------------------------------------------------
# amortized types
# ---------------------------------------------------------------------------
@app.command()
def types() -> None:
    """List available job types."""
    with _client() as client:
        resp = client.get("/api/v1/job-types")
        data = _handle_response(resp)

    table = Table(title="Job Types")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for jt in data:
        table.add_row(jt.get("name", ""), jt.get("description", ""))
    console.print(table)


# ---------------------------------------------------------------------------
# amortized recipes
# ---------------------------------------------------------------------------
@app.command()
def recipes(
    job_type: Annotated[str | None, typer.Option("--type", help="Filter by type")] = None,
) -> None:
    """List available recipes."""
    with _client() as client:
        resp = client.get("/api/v1/recipes")
        data = _handle_response(resp)

    if job_type:
        data = [r for r in data if r.get("type") == job_type]

    table = Table(title="Recipes")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description")

    for r in data:
        table.add_row(r.get("name", ""), r.get("type", ""), r.get("description", ""))
    console.print(table)


# ---------------------------------------------------------------------------
# amortized recipe <name>
# ---------------------------------------------------------------------------
@app.command()
def recipe(
    name: Annotated[str, typer.Argument(help="Recipe name")],
) -> None:
    """Show recipe details."""
    with _client() as client:
        resp = client.get(f"/api/v1/recipes/{name}")
        data = _handle_response(resp)

    console.print(Panel(
        json.dumps(data, indent=2),
        title=f"Recipe: {name}",
        border_style="blue",
    ))


# ---------------------------------------------------------------------------
# amortized artifacts
# ---------------------------------------------------------------------------
@app.command()
def artifacts(
    artifact_type: Annotated[
        str | None, typer.Option("--type", help="Filter by artifact type")
    ] = None,
    job_id: Annotated[str | None, typer.Option("--job-id", help="Filter by job ID")] = None,
) -> None:
    """List artifacts."""
    with _client() as client:
        params: dict[str, str] = {}
        if artifact_type:
            params["type"] = artifact_type
        if job_id:
            params["producer_job"] = job_id
        resp = client.get("/api/v1/artifacts", params=params)
        data = _handle_response(resp)

    table = Table(title="Artifacts")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Type", style="magenta")
    table.add_column("Job", style="dim")

    for a in data:
        table.add_row(
            a.get("id", ""),
            a.get("name", ""),
            a.get("artifact_type", a.get("type", "")),
            a.get("producer_job", a.get("job_id", "")),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# amortized backends
# ---------------------------------------------------------------------------
@app.command()
def backends() -> None:
    """List compute backends."""
    with _client() as client:
        resp = client.get("/api/v1/compute")
        data = _handle_response(resp)

    table = Table(title="Compute Backends")
    table.add_column("Name", style="cyan")
    table.add_column("Capabilities")

    for b in data:
        caps = ", ".join(b.get("capabilities", []))
        table.add_row(b.get("name", ""), caps)
    console.print(table)


# ---------------------------------------------------------------------------
# amortized health
# ---------------------------------------------------------------------------
@app.command()
def mcp(
    host: Annotated[str, typer.Option(help="Bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
) -> None:
    """Start MCP server (HTTP transport, auto-generated from OpenAPI)."""
    console.print(f"[bold green]Starting Amortized MCP server[/bold green] on {host}:{port}")
    console.print(f"MCP endpoint: http://{host}:{port}/mcp")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "amortized.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        check=False,
    )


@app.command()
def health() -> None:
    """Check API server health."""
    try:
        with _client() as client:
            resp = client.get("/api/v1/health")
            data = _handle_response(resp)
    except httpx.ConnectError:
        err_console.print("[red]Could not connect to the API server[/red]")
        raise typer.Exit(1) from None

    status = data.get("status", "unknown")
    color = "green" if status == "ok" else "red"
    console.print(f"[bold {color}]{status.upper()}[/bold {color}]  {_discover_url()}")

    gpu = data.get("gpu", {})
    if gpu.get("available"):
        devices = gpu.get("devices", [])
        dev_str = ", ".join(devices) or "detected"
        console.print(f"  GPU: {gpu.get('count', 0)} device(s) — {dev_str}")
    else:
        console.print("  GPU: [dim]not available[/dim]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "pending": "yellow",
    "queued": "yellow",
    "validating": "yellow",
    "provisioning": "cyan",
    "running": "blue",
    "completed": "green",
    "succeeded": "green",
    "failed": "red",
    "cancelled": "dim",
}


def _colorize_status(status: str) -> str:
    color = _STATUS_COLORS.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _print_job_panel(data: dict[str, Any]) -> None:
    lines = [
        f"[bold]ID:[/bold]      {data.get('id', '')}",
        f"[bold]Type:[/bold]    {data.get('type', '')}",
        f"[bold]Status:[/bold]  {_colorize_status(data.get('status', ''))}",
        f"[bold]Created:[/bold] {data.get('created_at', '')}",
    ]
    if data.get("error"):
        lines.append(f"[bold]Error:[/bold]   [red]{data['error']}[/red]")
    if data.get("config"):
        lines.append(f"[bold]Config:[/bold]  {json.dumps(data['config'], indent=2)}")
    if data.get("metadata"):
        lines.append(f"[bold]Meta:[/bold]    {json.dumps(data['metadata'])}")
    console.print(Panel("\n".join(lines), title="Job", border_style="cyan"))


def _print_event(event: dict[str, Any]) -> None:
    etype = event.get("type", "event")
    ts = event.get("timestamp", "")
    edata = event.get("data", {})
    msg = edata.get("message", json.dumps(edata) if edata else "")
    console.print(f"[dim]{ts}[/dim]  [{_STATUS_COLORS.get(etype, 'white')}]{etype}[/]  {msg}")
