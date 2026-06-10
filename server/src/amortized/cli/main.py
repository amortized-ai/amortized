"""Typer CLI — thin REST client wrapping the Amortized API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="amortized",
    help="Amortized — build task models that replace frontier API calls",
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
# amortized config
# ---------------------------------------------------------------------------


def _test_ssh(host: str, user: str | None = None) -> dict[str, Any]:
    """Test SSH connectivity and detect GPUs."""
    result: dict[str, Any] = {"connected": False}
    try:
        ssh_target = f"{user}@{host}" if user else host
        cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            ssh_target,
        ]
        gpu_cmd = [
            *cmd,
            "nvidia-smi --query-gpu=name,memory.total"
            " --format=csv,noheader,nounits 2>/dev/null"
            " || echo no-gpu",
        ]
        proc = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            result["connected"] = True
            output = proc.stdout.strip()
            if output and output != "no-gpu":
                gpus = [line.strip() for line in output.splitlines() if line.strip()]
                result["gpus"] = gpus
                result["gpu_count"] = len(gpus)
        else:
            result["error"] = proc.stderr.strip() or "Connection failed"
    except subprocess.TimeoutExpired:
        result["error"] = "Connection timed out"
    except FileNotFoundError:
        result["error"] = "ssh command not found"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _detect_container_runtime() -> str | None:
    """Detect Docker or Podman."""
    for runtime in ("podman", "docker"):
        try:
            proc = subprocess.run([runtime, "version"], capture_output=True, timeout=5)
            if proc.returncode == 0:
                return runtime
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _configure_ssh_backend() -> dict[str, Any] | None:
    """Interactive SSH backend configuration."""
    host = typer.prompt("SSH host")
    user = typer.prompt("SSH user (Enter for default)", default="", show_default=False)
    user = user or None

    console.print("Testing connection...", end=" ")
    ssh_result = _test_ssh(host, user)
    if ssh_result["connected"]:
        console.print("[green]✓ Connected[/green]")
        if ssh_result.get("gpus"):
            gpu_summary = f"{ssh_result['gpu_count']}x {ssh_result['gpus'][0]}"
            console.print(f"Detecting GPUs... [cyan]{gpu_summary}[/cyan]")
        else:
            console.print("Detecting GPUs... [dim]none detected[/dim]")
    else:
        console.print(f"[red]✗ {ssh_result.get('error', 'Failed')}[/red]")
        if not typer.confirm("Continue anyway?", default=False):
            return None

    # Detect container runtime on the REMOTE node (not locally)
    ssh_target = f"{user}@{host}" if user else host
    console.print("Detecting container runtime...", end=" ")
    container_rt = None
    for rt in ("podman", "docker"):
        detect_cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            ssh_target,
            f"{rt} --version",
        ]
        try:
            proc = subprocess.run(detect_cmd, capture_output=True, text=True, timeout=10)
            if proc.returncode == 0:
                container_rt = rt
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    if container_rt:
        console.print(f"[cyan]{container_rt}[/cyan]")
    else:
        console.print("[dim]none detected[/dim]")

    backend_name = typer.prompt("Name this backend", default="gpu-node")

    backend_cfg: dict[str, Any] = {"type": "ssh", "host": host}
    if user:
        backend_cfg["user"] = user
    backend_cfg["container_runtime"] = container_rt or "podman"
    console.print("\nHow should jobs execute on this node?")
    console.print("  [1] Docker / Podman container (recommended)")
    console.print("  [2] Bare metal (Python venv)")
    exec_choice = typer.prompt("Choice", default="1")
    if exec_choice == "2":
        backend_cfg["bare_metal"] = True
    else:
        rt = container_rt or "docker"
        console.print(f"\n[bold]Pull container images on {host}:[/bold]")
        console.print(f"  ssh {ssh_target} '{rt} pull ghcr.io/amortized-ai/training:latest'")
        console.print(f"  ssh {ssh_target} '{rt} pull ghcr.io/amortized-ai/synth:latest'")

    if ssh_result.get("gpus"):
        backend_cfg["gpu_info"] = ssh_result["gpus"]

    return {"name": backend_name, "config": backend_cfg}


def _discover_k8s_contexts() -> list[str]:
    """Read available K8s contexts from ~/.kube/config."""
    kube_config = Path.home() / ".kube" / "config"
    if not kube_config.exists():
        return []
    try:
        import yaml

        data = yaml.safe_load(kube_config.read_text())
        if isinstance(data, dict) and "contexts" in data:
            return [ctx["name"] for ctx in data["contexts"] if isinstance(ctx, dict)]
    except Exception:
        pass
    return []


def _configure_k8s_backend() -> dict[str, Any] | None:
    """Interactive K8s backend configuration."""
    contexts = _discover_k8s_contexts()
    if contexts:
        console.print("\nKubernetes contexts (from ~/.kube/config):")
        for i, ctx in enumerate(contexts, 1):
            console.print(f"  [{i}] {ctx}")
        console.print(f"  [{len(contexts) + 1}] Enter manually")
        choice = typer.prompt("Choice", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(contexts):
                context = contexts[idx]
            else:
                context = typer.prompt("K8s context name")
        except ValueError:
            context = choice
    else:
        context = typer.prompt("K8s context name")

    namespace = typer.prompt("Namespace", default="default")
    backend_name = typer.prompt("Name this backend", default=context)

    return {
        "name": backend_name,
        "config": {
            "type": "kubernetes",
            "context": context,
            "namespace": namespace,
        },
    }


@app.command()
def config() -> None:
    """Configure compute runtimes for job dispatch."""
    import yaml

    config_path = Path.home() / ".amortized" / "config.yaml"

    existing: dict[str, Any] = {}
    if config_path.exists():
        existing = yaml.safe_load(config_path.read_text()) or {}

    backends: dict[str, dict[str, Any]] = dict(existing.get("compute", {}).get("backends", {}))

    if backends:
        console.print("\n[bold]Current backends:[/bold]")
        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Host / Context")
        for name, bcfg in backends.items():
            host = bcfg.get("host", bcfg.get("context", ""))
            table.add_row(name, bcfg.get("type", ""), host)
        console.print(table)

        console.print("\nWhat would you like to do?")
        console.print("  [1] Add a new backend")
        console.print("  [2] Remove a backend")
        console.print("  [3] Reconfigure from scratch")
        console.print("  [4] Done")
        action = typer.prompt("Choice", default="1")

        if action == "2":
            name = typer.prompt("Backend name to remove")
            if name in backends:
                del backends[name]
                console.print(f"[yellow]Removed {name}[/yellow]")
            else:
                console.print(f"[red]Not found: {name}[/red]")
            existing.setdefault("compute", {})["backends"] = backends
            config_path.write_text(yaml.dump(existing, default_flow_style=False))
            console.print(f"[green]✓ Saved to {config_path}[/green]")
            return
        elif action == "3":
            backends = {}
        elif action == "4":
            return

    new_backends: dict[str, dict[str, Any]] = {}

    while True:
        console.print("\n[bold]Where will your jobs run?[/bold]")
        console.print("  [1] This machine (local)")
        console.print("  [2] Remote GPU node (SSH)")
        console.print("  [3] Kubernetes / OpenShift cluster")
        console.print("  [4] Skip — configure later")
        choice = typer.prompt("Choice", default="1")

        if choice == "1":
            console.print("[green]✓ Local backend always available[/green]")
        elif choice == "2":
            result = _configure_ssh_backend()
            if result:
                new_backends[result["name"]] = result["config"]
                console.print(f"[green]✓ Added backend '{result['name']}'[/green]")
        elif choice == "3":
            result = _configure_k8s_backend()
            if result:
                new_backends[result["name"]] = result["config"]
                console.print(f"[green]✓ Added backend '{result['name']}'[/green]")
        elif choice == "4":
            break
        else:
            console.print("[yellow]Invalid choice[/yellow]")
            continue

        if not typer.confirm("\nAdd another backend?", default=False):
            break

    backends.update(new_backends)
    existing.setdefault("compute", {})["backends"] = backends

    if len(backends) == 1:
        default_name = next(iter(backends.keys()))
        existing["compute"]["default_backend"] = default_name
        console.print(f"[dim]Default backend set to '{default_name}'[/dim]")
    elif len(backends) > 1:
        console.print("\nWhich backend should be the default for GPU jobs?")
        for i, name in enumerate(backends, 1):
            console.print(f"  [{i}] {name}")
        choice = typer.prompt("Choice", default="1")
        try:
            idx = int(choice) - 1
            default_name = list(backends.keys())[idx]
        except (ValueError, IndexError):
            default_name = next(iter(backends.keys()))
        existing["compute"]["default_backend"] = default_name

    # Auto-detect external_url for SSH backends so remote nodes can reach us
    has_ssh = any(b.get("type") == "ssh" for b in backends.values())
    if has_ssh and not existing.get("external_url"):
        import socket

        local_hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(local_hostname)
        except socket.gaierror:
            local_ip = None
        if local_ip and local_ip != "127.0.0.1":
            console.print(f"Control plane reachable at: [cyan]http://{local_ip}:8000[/cyan]")
            existing["external_url"] = f"http://{local_ip}:8000"

    console.print("\n[bold]Credentials[/bold]")
    detected: dict[str, str] = {}
    for env_var, value in sorted(os.environ.items()):
        if any(env_var.endswith(suffix) for suffix in ("_API_KEY", "_TOKEN", "_SECRET_KEY")):
            preview = f"...{value[-4:]}" if len(value) >= 4 else "***"
            detected[env_var] = preview

    if detected:
        console.print("Detected credentials in your environment:")
        items = list(detected.items())
        for i, (name, preview) in enumerate(items, 1):
            console.print(f"  [{i}] {name:30s} {preview}")

        selection = typer.prompt(
            "Select which to forward to remote jobs (comma-separated, 'all', or 'none')",
            default="all",
        )

        if selection.lower() == "all":
            selected = [name for name, _ in items]
        elif selection.lower() == "none":
            selected = []
        else:
            selected = []
            all_names = {name for name, _ in items}
            for part in selection.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(items):
                        selected.append(items[idx][0])
                elif part in all_names:
                    selected.append(part)

        if selected:
            existing["forward_env"] = selected
            console.print(f"[green]✓ {len(selected)} credentials will be forwarded[/green]")
    else:
        console.print("[dim]No credentials detected (set env vars like OPENAI_API_KEY)[/dim]")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(existing, default_flow_style=False))

    console.print(f"\n[green]✓ Saved to {config_path}[/green]")

    if backends:
        table = Table(title="Configured Backends")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Host / Context")
        table.add_column("GPUs", style="dim")
        for name, bcfg in backends.items():
            host_or_ctx = bcfg.get("host", bcfg.get("context", ""))
            gpu_info = ""
            if bcfg.get("gpu_info"):
                gpu_info = f"{len(bcfg['gpu_info'])}x {bcfg['gpu_info'][0]}"
            table.add_row(name, bcfg["type"], host_or_ctx, gpu_info)
        console.print(table)

    console.print("\nRun [bold]amortized up[/bold] to start the server.")


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def _resolve_data(client: httpx.Client, value: str) -> str:
    """Validate an artifact ID or return a file path as-is.

    If the value is a UUID, validates the artifact exists and returns an
    ``artifact:<id>`` reference for the worker to resolve at dispatch time.
    """
    if not _is_uuid(value):
        return value
    resp = client.get(f"/api/v1/artifacts/{value}")
    if resp.status_code != 200:
        err_console.print(f"[red]Artifact {value} not found[/red]")
        raise typer.Exit(1)
    name = resp.json().get("name", value)
    console.print(f"[dim]Using artifact: {name} ({value[:8]}…)[/dim]")
    return f"artifact:{value}"


# ---------------------------------------------------------------------------
# amortized submit
# ---------------------------------------------------------------------------
@app.command()
def submit(
    job_type: Annotated[str, typer.Argument(help="Job type (e.g. training, sdg)")],
    config: Annotated[str | None, typer.Option("--config", help="JSON config string")] = None,
    recipe: Annotated[str | None, typer.Option("--recipe", help="Recipe name")] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model path (training) or model name (sdg)"),
    ] = None,
    data: Annotated[str | None, typer.Option("--data", help="Data path or artifact ID")] = None,
    set_values: Annotated[
        list[str] | None, typer.Option("--set", help="Override KEY=VALUE")
    ] = None,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Actually submit (default is dry-run preview)")
    ] = False,
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
                if not k.startswith("config."):
                    k = f"config.{k}"
                try:
                    overrides[k] = json.loads(v)
                except json.JSONDecodeError:
                    overrides[k] = v
            model_key = "model" if job_type == "sdg" else "model_name_or_path"
            if model:
                overrides.setdefault(f"config.{model_key}", model)
            if data:
                overrides.setdefault("config.data_path", _resolve_data(client, data))
            for k in list(overrides):
                v = overrides[k]
                bare = k.removeprefix("config.")
                if isinstance(v, str) and _is_uuid(v) and bare.endswith("_path"):
                    overrides[k] = _resolve_data(client, v)
            body: dict[str, Any] = {
                "recipe": recipe,
                "overrides": overrides,
                "dry_run": not confirm,
            }
            resp = client.post("/api/v1/jobs/recipe", json=body)
        else:
            cfg: dict[str, Any] = {}
            if config:
                try:
                    cfg = json.loads(config)
                except json.JSONDecodeError as exc:
                    err_console.print(f"[red]Invalid JSON config:[/red] {exc}")
                    raise typer.Exit(1) from exc
            model_key = "model" if job_type == "sdg" else "model_name_or_path"
            if model:
                cfg.setdefault(model_key, model)
            if data:
                cfg.setdefault("data_path", _resolve_data(client, data))
            for kv in set_values or []:
                if "=" not in kv:
                    err_console.print(f"[red]Invalid --set format:[/red] {kv} (expected KEY=VALUE)")
                    raise typer.Exit(1)
                k, v = kv.split("=", 1)
                try:
                    cfg[k] = json.loads(v)
                except json.JSONDecodeError:
                    cfg[k] = v

            body = {"type": job_type, "config": cfg, "dry_run": not confirm}
            resp = client.post("/api/v1/jobs", json=body)

        resp_data = _handle_response(resp)
        if "valid" in resp_data and "id" not in resp_data:
            if resp_data["valid"]:
                console.print("[green]✓ Config valid[/green]")
            else:
                console.print("[red]✗ Config invalid[/red]")
            for err in resp_data.get("errors", []):
                console.print(f"  [red]• {err}[/red]")
            if resp_data.get("warnings"):
                for w in resp_data["warnings"]:
                    console.print(f"  [yellow]⚠ {w}[/yellow]")
            console.print("\nRun with [bold]--confirm[/bold] to submit.")
            return
        _print_job_panel(resp_data)


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
        table.add_row(jt.get("type", ""), jt.get("description", ""))
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

    console.print(
        Panel(
            json.dumps(data, indent=2),
            title=f"Recipe: {name}",
            border_style="blue",
        )
    )


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
# amortized upload
# ---------------------------------------------------------------------------
@app.command()
def upload(
    file_path: Annotated[str, typer.Argument(help="Local file to upload as an artifact")],
    artifact_type: Annotated[str, typer.Option("--type", "-t", help="Artifact type")] = "dataset",
    name: Annotated[str | None, typer.Option("--name", "-n", help="Artifact name")] = None,
) -> None:
    """Upload a local file as an artifact."""
    p = Path(file_path)
    if not p.is_file():
        err_console.print(f"[red]File not found:[/red] {file_path}")
        raise typer.Exit(1)

    artifact_name = name or p.name
    with _client() as client:
        with p.open("rb") as f:
            resp = client.post(
                "/api/v1/artifacts/upload",
                files={"file": (p.name, f)},
                data={"artifact_type": artifact_type, "name": artifact_name},
            )
        data = _handle_response(resp)

    console.print(
        Panel(
            f"[bold]ID:[/bold]   {data.get('id', '')}\n"
            f"[bold]Name:[/bold] {data.get('name', '')}\n"
            f"[bold]Type:[/bold] {data.get('artifact_type', '')}\n"
            f"[bold]Path:[/bold] {data.get('location', '')}",
            title="Uploaded Artifact",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# amortized mcp
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
