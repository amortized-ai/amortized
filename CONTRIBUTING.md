# Contributing to Amortized

Thanks for your interest in contributing to Amortized!

## Getting started

1. Fork the repository
2. Clone your fork: `git clone git@github.com:<your-username>/amortized.git`
3. Install dependencies:
   ```bash
   uv pip install -e '.[dev]'
   cd studio && npm install
   ```
4. Create a branch: `git checkout -b feat/my-feature`

## Development workflow

### Backend

```bash
amortized up              # start server on :8000
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
mypy src/                 # type check
pytest tests/ -x -q       # test
```

### Frontend

```bash
cd studio
npm run dev               # dev server on :5173
npm run lint              # lint
npm run typecheck         # type check
npm test                  # test
```

## Pull requests

- Keep PRs focused on a single concern
- Write a clear title (under 70 characters) and description
- Include a test plan or describe how changes were verified
- Make sure CI passes (lint, typecheck, tests)

## Code style

- Python: ruff for linting/formatting, mypy for type checking
- TypeScript: ESLint + Prettier via npm scripts
- Follow existing patterns in the codebase

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add dataset upload endpoint
fix: handle empty config in training job
docs: update deployment guide
```

## Reporting issues

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Environment details (OS, Python version, etc.)

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
