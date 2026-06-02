# Factory Configuration — Amortized

## Project

- **Name**: Amortized
- **Type**: product
- **Repository**: amortized-ai/amortized

## Eval Dimensions

| Dimension | Script | Description |
|---|---|---|
| tests | `eval/score.py` | Run pytest in runtime/ and check exit code |
| lint | `eval/score.py` | Run ruff check in runtime/ |
| type_check | `eval/score.py` | Run mypy in runtime/ |
| capability_surface | `eval/score.py` | Count API endpoints defined |
| observability | `eval/score.py` | Check for logging setup |

## Mutable Surfaces

- `CLAUDE.md`
- `factory.md`
- `runtime/src/**`
- `runtime/tests/**`
- `runtime/pyproject.toml`
- `studio/src/**`
- `studio/package.json`
- `studio/tsconfig.json`
- `studio/next.config.mjs`
- `studio/tailwind.config.ts`
- `studio/postcss.config.mjs`
- `docker/**`
- `.github/**`

## Fixed Surfaces

- `eval/score.py`
- `.factory/**`
- `LICENSE`
- `README.md`
