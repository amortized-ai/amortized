# Amortized Studio

Web UI for [Amortized](https://github.com/amortized-ai/amortized) — the control plane for building task-specific AI models on OpenShift. Studio provides a guided interface for synthetic data generation, model fine-tuning, and evaluation, with an AI chat assistant (Morty) that walks you through the entire workflow.

## Features

- **Chat assistant (Morty)** — Agentic chat with tool use. Morty can list jobs, submit recipes, check logs, and guide you step-by-step through SDG, training, and eval workflows. Post-submission option cards let you navigate to results or continue to the next step.
- **Recipe builder** — Visual form for creating training, SDG, and eval recipes with method selection, model/dataset pickers, and hyperparameter controls. Supports JSON editor for advanced configuration.
- **Job management** — Table view of all jobs with status filters, search, and detail panels showing overview, logs, metrics, and config tabs. Editable job titles.
- **Dataset viewer** — Browse MLflow-tracked datasets with a samples tab that renders JSONL content (handles malformed JSON and object-typed message content).
- **Model registry** — View registered models and their versions from MLflow.
- **Settings** — Platform configuration display, AI Gateway route management (add/delete LLM provider endpoints for SDG and eval), and API key storage.

## Prerequisites

- **Node.js 22+** and npm
- **Amortized backend** running on `localhost:8000` — see the [amortized repo](https://github.com/amortized-ai/amortized) for setup
- **MLflow** running on `localhost:5000` (optional — needed for datasets, models, and metrics)

## Development

```bash
# Install dependencies
npm install

# Start dev server (proxies /api to localhost:8000, /mlflow to localhost:5000)
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

The Vite dev server proxies API requests to the backend automatically:

| Path | Target |
|------|--------|
| `/api/*` | `http://localhost:8000` (Amortized backend) |
| `/mlflow/*` | `http://localhost:5000` (MLflow) |
| `/agent/*` | `http://localhost:8000` (Chat agent) |

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Override the backend API base URL | `""` (uses proxy) |

### Commands

```bash
npm run dev          # Start dev server with hot reload
npm run build        # Type-check and build for production
npm run preview      # Preview production build locally
npm run test         # Run unit tests (vitest)
npm run test:watch   # Run tests in watch mode
npm run test:e2e     # Run Playwright end-to-end tests
npm run lint         # Lint with ESLint
npm run typecheck    # Type-check with TypeScript
npm run format       # Format with Prettier
```

## Production build

```bash
npm run build
```

Output goes to `dist/`. Serve it with any static file server. In production, the `Dockerfile.kind` builds an nginx image that serves the static files and proxies API requests. See `nginx.conf.template` for the proxy configuration — it uses environment variable substitution for backend hosts:

| Variable | Description |
|----------|-------------|
| `BACKEND_HOST` | Amortized backend hostname |
| `BACKEND_PORT` | Amortized backend port |
| `AGENT_HOST` | OpenCode/Claude Code agent hostname |
| `AGENT_PORT` | Agent port |
| `MLFLOW_HOST` | MLflow server hostname |
| `MLFLOW_PORT` | MLflow server port |
| `DNS_RESOLVER` | DNS resolver for nginx (e.g., `10.96.0.10` for kube-dns) |

## Cluster deployment

Studio is deployed on a kind cluster with per-developer namespaces. Each developer gets their own Studio instance backed by their own Amortized server. See the [amortized repo](https://github.com/amortized-ai/amortized) README for the full port mapping table and deployment commands.

Access via SSH tunnel (forward your studio port):

```bash
ssh -L <your-studio-port>:localhost:<your-studio-port> <user>@<gpu-host>
```

Each developer works from their own directory on the cluster (`~/<username>/studio/`). See the [amortized repo](https://github.com/amortized-ai/amortized) README for full directory layout and setup.

### Testing a PR branch

Check out the branch in your studio clone, then rebuild and redeploy from your amortized directory:

```bash
# On the cluster
cd ~/<username>/studio
git checkout feat/my-branch

cd ~/<username>/amortized
make refresh-<username>
```

To revert to main, check out main in your studio clone and run `make refresh-<username>` again.

## Tech stack

- React 19, TypeScript, Vite
- Tailwind CSS v4, shadcn/ui (Radix primitives)
- TanStack Query (data fetching), TanStack Table
- Zustand (client state), React Router v7
- Recharts (metrics visualization)
- CodeMirror (JSON editor)
- Vitest + Testing Library (unit tests), Playwright (E2E)
