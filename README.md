# amortized

*Build task models that replace frontier API calls*

---

Every AI agent has tasks that don't need a frontier model. Classification, extraction, routing, summarization — these are specific, repeatable, and learnable. A small fine-tuned model can do them faster, cheaper, and more reliably than a general-purpose API.

**Amortized builds these task models.** You describe the task, it generates training data from a teacher model, fine-tunes a small student model, and evaluates whether the student matches the teacher. The result: a model you own that runs on your infrastructure, costs a fraction per inference, and doesn't break when the API provider changes.

The name comes from finance — amortization spreads a large upfront cost across many future uses. Here, the "cost" is the frontier model's capability, and the "uses" are every future inference by the cheaper task model.

## Deployment

Amortized runs on a kind cluster with GPU passthrough. Each developer gets an isolated namespace with their own server, studio, OpenCode, and Claude Code deployments. Shared services (MLflow, MinIO) run in the `amortized` namespace.

### Developer environments

| Developer | Namespace | Studio | Server | GPU Quota |
|-----------|-----------|--------|--------|-----------|
| meyceoz   | amortized-meyceoz  | 31100 | 31101 | 1 |
| ssudalai  | amortized-ssudalai | 31110 | 31111 | 1 |
| mathale   | amortized-mathale  | 31120 | 31121 | 1 |
| nmalepat  | amortized-nmalepat | 31130 | 31131 | 1 |
| esivaram  | amortized-esivaram | 31140 | 31141 | 1 |
| *(shared)* | amortized         | —     | MLflow: 31082 | — |

### Quick start

```bash
# First-time setup (creates cluster, configures GPUs, deploys everything)
make up GHCR_USER=<github-user> GHCR_TOKEN=<github-pat>

# Deploy shared services only (MLflow, MinIO)
make deploy-shared

# Deploy your environment
make deploy-<username>

# Deploy all environments
make deploy-all

# Rebuild images and redeploy
make refresh-<username>

# Tear down your environment
make down-<username>

# Check status
make status
```

### Access

```bash
ssh -L 31082:localhost:31082 \
    -L 31100:localhost:31100 -L 31101:localhost:31101 \
    -L 31110:localhost:31110 -L 31111:localhost:31111 \
    -L 31120:localhost:31120 -L 31121:localhost:31121 \
    -L 31130:localhost:31130 -L 31131:localhost:31131 \
    -L 31140:localhost:31140 -L 31141:localhost:31141 \
    <user>@<gpu-host>
```

Then open your studio at `http://localhost:<your-studio-port>` (see table above for port assignments).

### Testing a PR branch

To deploy a custom studio image for PR testing:

```bash
# Build from a different studio directory
docker build -t amortized-studio:my-pr -f ../<username>/studio/Dockerfile.kind ../<username>/studio/
kind load docker-image amortized-studio:my-pr --name amortized

# Switch your deployment to the custom image
kubectl -n amortized-<username> set image deployment/amortized-studio studio=amortized-studio:my-pr
```

### Adding a new developer

1. Create a new overlay directory under `k8s/overlays/users/<username>/` (copy from an existing user and update the namespace, ports, and user references)
2. Pick unused NodePorts (next available: studio=31150, server=31151)
3. Add the port mappings to `k8s/kind/kind-config.yaml`
4. Add the username to `USERS` in `Makefile`
5. Run `make deploy-<newuser>`

## License

[Apache 2.0](LICENSE)
