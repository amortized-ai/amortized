# amortized

*Build task models that replace frontier API calls*

---

Every AI agent has tasks that don't need a frontier model. Classification, extraction, routing, summarization — these are specific, repeatable, and learnable. A small fine-tuned model can do them faster, cheaper, and more reliably than a general-purpose API.

**Amortized builds these task models.** You describe the task, it generates training data from a teacher model, fine-tunes a small student model, and evaluates whether the student matches the teacher. The result: a model you own that runs on your infrastructure, costs a fraction per inference, and doesn't break when the API provider changes.

The name comes from finance — amortization spreads a large upfront cost across many future uses. Here, the "cost" is the frontier model's capability, and the "uses" are every future inference by the cheaper task model.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Node.js 18+ and npm
- A Kubernetes cluster (for deployment)
- PostgreSQL
- MLflow (for artifact tracking)
- S3-compatible storage (e.g., MinIO)

## Repository layout

```
amortized/
├── src/amortized/       # Python backend (FastAPI)
├── studio/              # React frontend (Vite)
├── agent/               # Agent prompts and skills
├── containers/          # Training container Dockerfiles
├── templates/           # YAML templates for SDG/training
├── k8s/                 # Kubernetes manifests (kustomize)
├── docs/                # Architecture docs and ADRs
└── Makefile             # Build targets
```

## Getting started

### Backend

```bash
# Install dependencies
uv pip install -e '.[dev]'

# Configure compute backend
amortized config

# Copy and edit environment variables
cp .env.example .env

# Start the server
amortized up
```

The server starts on `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### Studio (frontend)

```bash
cd studio
npm install
npm run dev
```

The dev server starts on `http://localhost:5173`.

### Linting and testing

```bash
# Backend
ruff check src/ tests/
ruff format src/ tests/
mypy src/
pytest tests/ -x -q

# Frontend
cd studio
npm run lint
npm run typecheck
npm test
```

## Kubernetes deployment

Amortized ships with kustomize overlays for Kubernetes deployment.

### Development (single-user)

Deploys the server, studio, MLflow, MinIO, and PostgreSQL:

```bash
kubectl apply -k k8s/overlays/dev
```

Or use the Makefile shortcut:

```bash
make deploy-dev
```

### OpenShift / ROSA

```bash
kubectl apply -k k8s/overlays/rosa
```

### Building container images

```bash
make build            # Build server + studio images
make build-server     # Build server image only
make build-studio     # Build studio image only
```

## Configuration

All configuration is via environment variables with the `AMORTIZED_` prefix. See [`.env.example`](.env.example) for the full list.

Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `AMORTIZED_DATABASE_URL` | PostgreSQL connection string | `postgresql://amortized:amortized@localhost:5432/amortized` |
| `AMORTIZED_COMPUTE_BACKEND` | Compute backend (`local`, `ssh`, `kubernetes`) | `local` |
| `AMORTIZED_MLFLOW_TRACKING_URI` | MLflow tracking server URL | (empty = disabled) |
| `AMORTIZED_S3_BUCKET` | S3 bucket for artifacts | (empty) |
| `AMORTIZED_IMAGE_REGISTRY` | Container image registry for jobs | `ghcr.io/amortized-ai` |

## Architecture

Amortized is a thin orchestration layer. It translates user intent into tool-native YAML configs, dispatches K8s Jobs, and tracks job lifecycle. Everything else is delegated: MLflow for artifacts/lineage, S3 for storage, K8s for compute, TRL/asynth/vLLM for ML logic.

Two job types:
- **Training** — Fine-tunes models using TRL or training-hub
- **SDG** — Generates synthetic training data using Data Designer

See [`docs/architecture.md`](docs/architecture.md) for the full system description.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache 2.0](LICENSE)
