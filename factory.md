# Factory Configuration — Amortized

## Project

- **Name**: Amortized
- **Type**: product
- **Repository**: amortized-ai/amortized

## Eval Dimensions

| Dimension | Script | Description |
|---|---|---|
| tests | `eval/score.py` | Run pytest in server/ and check exit code |
| lint | `eval/score.py` | Run ruff check in server/ |
| type_check | `eval/score.py` | Run mypy in server/ |
| capability_surface | `eval/score.py` | Count API endpoints defined |
| observability | `eval/score.py` | Check for logging setup |

## Mutable Surfaces

- `CLAUDE.md`
- `factory.md`
- `server/src/**`
- `server/tests/**`
- `server/pyproject.toml`
- `studio/src/**`
- `studio/package.json`
- `studio/tsconfig.json`
- `studio/next.config.mjs`
- `studio/tailwind.config.ts`
- `studio/postcss.config.mjs`
- `containers/**`
- `docker/**`
- `.github/**`

## Fixed Surfaces

- `eval/score.py`
- `.factory/**`
- `LICENSE`
- `README.md`

## Goal

Build a fully open-source, on-premises AI model customization studio that replaces expensive frontier model calls with smaller, customized models. Two core components: a containerized runtime API (FastAPI + Training Hub + SDG Hub) and a studio UI (Next.js) with agent-guided experience.

## Guards

- Do not remove existing API endpoints
- Do not break the health check endpoint
- Do not hardcode API keys or secrets
- Do not modify eval/score.py

## Eval Command

```
python eval/score.py
```

## Threshold

0.75

## Smoke Test

```
curl -sf http://localhost:8000/api/v1/health
```
