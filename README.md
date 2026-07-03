# amortized

*Build task models that replace frontier API calls*

---

A control plane for building small, fine-tuned models that replace expensive frontier model calls for specific tasks (classification, extraction, routing, summarization). Deployed on OpenShift.

**SDG → Training → Eval** — each step is a K8s Job. MLflow tracks all artifacts. [Studio](https://github.com/amortized-ai/studio) provides the UI.

## Quick Start

```bash
uv pip install -e '.[dev]'
amortized config   # configure compute backend
amortized up       # start server on :8000
```

## Docs

- [Architecture](docs/architecture/control-plane.md)
- [Architecture Decisions](docs/architecture/adr-001-control-plane.md)

## License

[Apache 2.0](LICENSE)
