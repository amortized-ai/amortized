# amortized

*Build task models that replace frontier API calls*

---

Every AI agent has tasks that don't need a frontier model. Classification, extraction, routing, summarization — these are specific, repeatable, and learnable. A small fine-tuned model can do them faster, cheaper, and more reliably than a general-purpose API.

**Amortized builds these task models.** You describe the task, it generates training data from a teacher model, fine-tunes a small student model, and evaluates whether the student matches the teacher. The result: a model you own that runs on your infrastructure, costs a fraction per inference, and doesn't break when the API provider changes.

The name comes from finance — amortization spreads a large upfront cost across many future uses. Here, the "cost" is the frontier model's capability, and the "uses" are every future inference by the cheaper task model.

## Deployment

Amortized runs on a kind cluster with GPU passthrough. See [docs/kind-setup.md](docs/kind-setup.md) for full setup instructions.

### Quick start

```bash
# First-time setup (creates cluster, configures GPUs, deploys everything)
make up GHCR_USER=<github-user> GHCR_TOKEN=<github-pat>

# Deploy prod (pulls images from GHCR)
make deploy GHCR_USER=<github-user> GHCR_TOKEN=<github-pat>

# Deploy dev (builds from local source)
make deploy-dev GHCR_USER=<github-user> GHCR_TOKEN=<github-pat>
```

### Access

```bash
ssh -L 31080:localhost:31080 -L 31081:localhost:31081 -L 31082:localhost:31082 \
    -L 31090:localhost:31090 -L 31091:localhost:31091 user@<gpu-node>
```

| Service | URL |
|---------|-----|
| Prod Studio | http://localhost:31080 |
| Prod API | http://localhost:31081 |
| MLflow | http://localhost:31082 |
| Dev Studio | http://localhost:31090 |
| Dev API | http://localhost:31091 |

## License

[Apache 2.0](LICENSE)
