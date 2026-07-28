# Factory Configuration
<!-- This file configures the Remote Factory for your project. -->
<!-- The factory reads this during Init mode and generates .factory/config.json from it. -->
<!-- Fill in each section below. -->

## Goal
<!-- A single sentence describing what this project should achieve. -->

A control plane for building task-specific fine-tuned models on OpenShift — translating user intent into tool-native YAML configs, dispatching K8s Jobs, and tracking job lifecycle across training, SDG, and eval workloads.

## Scope

### Modifiable
<!-- Files and directories the factory is allowed to create or edit. -->
<!-- One path per line. Glob patterns are supported. -->

- src/amortized/**/*.py
- tests/**/*.py
- studio/src/**/*.ts
- studio/src/**/*.tsx
- studio/src/**/*.css
- templates/**/*.yaml
- eval/**/*.py

### Read-only
<!-- Files the factory may read but must never modify. -->

- CLAUDE.md
- SPEC.md
- pyproject.toml
- studio/package.json
- docs/**/*
- k8s/**/*
- openapi/v1.json

## Guards
<!-- Rules the factory must never violate. Checked before every commit. -->

- Do not delete or overwrite existing tests
- Do not modify files outside the declared scope
- Do not introduce secrets or credentials into the repository
- Do not modify CLAUDE.md or SPEC.md
- Do not generate Python scripts for job dispatch — all job behavior must be defined by YAML configs
- Do not write directly to S3 — all artifacts must flow through MLflow
- Do not remove or weaken SSRF protection in document URL conversion
- Do not remove constant-time token comparison in authentication

## Eval

### Command
<!-- The shell command the factory runs to score a change. -->
<!-- It must output JSON to stdout matching the EvalResult format. -->

```bash
python eval/score.py
```

### Threshold
<!-- Minimum composite score (0.0-1.0) required to keep a change. -->

0.8

## Target Branch
<!-- Branch that experiment PRs target. Default: main -->

main

## Project Eval
<!-- User-defined project-specific eval dimensions (benchmarks, accuracy, latency, etc.) -->

## Eval Weights
<!-- Weight distribution across eval tiers (must sum to 1.0) -->
<!-- Default without project eval: hygiene 0.50, growth 0.50 -->

## Smoke Test
<!-- Optional shell command that must pass before any change is kept. -->

```bash
cd studio && npm run build && cd .. && python -c "from amortized.main import app; print('import ok')"
```

## Constraints
<!-- Soft rules that guide behavior but don't block commits. -->

- Prefer small, incremental changes over large rewrites
- Each change should be accompanied by at least one test
- Follow the existing code style and conventions (ruff, strict mypy, no comments unless WHY is non-obvious)
- Config-only dispatch — no generated Python scripts for jobs
- Single code path for all compute backends (K8s, SSH, local)
- MLflow is the artifact store — no direct S3 writes
- Use uv for Docker builds, not pip
- SQLite with raw aiosqlite — no ORM
- All API endpoints under /api/v1/

## Eval Spec
<!-- Discovered eval dimensions from .factory/eval_profile.json -->

- name: tests
  command: python -m pytest -v
  parse: exit_code
  weight: 0.4167
  description: Run test suite

- name: lint
  command: python -m ruff check .
  parse: exit_code
  weight: 0.25
  description: Run linter

- name: type_check
  command: python -m mypy ./
  parse: exit_code
  weight: 0.125
  description: Run type checker

- name: coverage
  command: python -m pytest --cov=src/amortized --cov-report=term -q
  parse: exit_code
  weight: 0.125
  description: Measure test coverage

- name: observability
  command: (inline)
  parse: json
  weight: 0.0833
  description: Analyze logging coverage, structured logging, and request tracing

## Research Target
<!-- Only for research/benchmark projects. Not applicable. -->

## Mutable Surfaces
<!-- Only used in research mode. Not applicable. -->

## Fixed Surfaces
<!-- Only used in research mode. Not applicable. -->

## Research Constraints
<!-- Only used in research mode. Not applicable. -->

## Cost Budget
<!-- Per-cycle or total budget constraints for research experiments. -->
