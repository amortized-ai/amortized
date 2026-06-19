# Amortized

## What We're Building

A product that builds task models. The user describes a task their AI agent does with a frontier model (classification, extraction, routing, summarization), and amortized produces a small fine-tuned model that does it cheaper, faster, and on their infrastructure.

This is not a toolkit or framework. It's an opinionated, batteries-included product. The user shouldn't configure training frameworks, pick container images, or wire up serving infrastructure. They describe the task, we handle everything else.

When in doubt, pick the simpler, more opinionated choice. One way to do things, documented in examples.

## Dev Commands

```bash
pip install -e '.[dev]'        # install
amortized up                   # start server on :8000
amortized config               # configure GPU backend
ruff check src/ tests/         # lint
ruff format src/ tests/        # format
mypy src/                      # type check
pytest tests/ -x -q            # test
python scripts/export_openapi.py  # regenerate openapi/v1.json
```

## Code Style

- Python 3.11+, strict mypy, ruff enforced
- `src/` layout — all code under `src/amortized/`
- FastAPI with pydantic-settings for config
- All API endpoints under `/api/v1/`
- SQLite for persistence (no ORM, raw aiosqlite)
- No comments unless the WHY is non-obvious

## Architecture

4 job types: `training`, `sdg`, `eval`, `serve`. Each maps to a container image:

- Training: `docker.io/huggingface/trl:1.5.0` — worker generates TRL YAML config
- SDG: `ghcr.io/amortized-ai/asynth` — worker generates YAML config, runs `asynth synthesize --config`
- Eval: `ghcr.io/amortized-ai/asynth` — worker generates Python script (deterministic checks + model inference), judge portion calls `asynth judge` CLI
- Serve: `docker.io/vllm/vllm-openai` — worker generates vLLM YAML config

No custom containers. The worker generates configs/scripts and the SSH backend writes them to the remote node, then runs the official image with the generated config as CMD override.

## Key Patterns

- **Config generation**: `_training_config_yaml()`, `_serve_config_yaml()`, `_build_synth_config()`, `_eval_script()` in `worker.py`
- **Artifact refs**: `artifact:<uuid>` in configs, resolved by worker to remote paths via `_resolve_artifact_refs()`
- **Training artifacts**: single directory-level "model" artifact per job (not per-file)
- **Judge templates**: loaded from `templates/eval/` by `core/judge_templates.py`, resolved at dispatch time by `_resolve_judge_template()`
- **Serve jobs**: long-running, background `_monitor_serve_job()` task, don't block worker loop
- **Recipes**: loaded from `templates/` and `examples/` via `core/recipes.py`, support `extends:` for inheritance
- **Credentials**: env var names in `~/.amortized/config.yaml` under `forward_env`, values read from os.environ at dispatch, injected via podman secrets

## Git

- Pre-commit hook regenerates `openapi/v1.json` if API files change
- `git config core.hooksPath .githooks`

## Gotchas

- `yaml.safe_load()` parses `2e-4` as string, not float — use `0.0002` in YAML configs
- vLLM image entrypoint is `["vllm", "serve"]` — container CMD should be just `["--config", "path"]`, not `["vllm", "serve", "--config"]`
- Eval containers need `--network host` to reach vLLM on the same node via `127.0.0.1`
- `~` in config paths doesn't expand inside containers — worker must resolve to absolute paths
- TRL field names differ from common conventions: `num_train_epochs` not `num_epochs`, `per_device_train_batch_size` not `batch_size`, `max_length` not `max_seq_len`
- asynth CLI config paths must be relative (e.g. `output/generated_data.jsonl`), not absolute container paths — the local backend runs from `cwd=work_dir`, containers mount work_dir at `/amortized/work`
