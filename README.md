<p align="center"><strong>amortized</strong></p>
<p align="center"><em>Build task models that replace frontier API calls</em></p>
<p align="center">
  <a href="https://github.com/amortized-ai/amortized/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/amortized-ai/amortized/ci.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://github.com/amortized-ai/amortized/blob/main/LICENSE"><img src="https://img.shields.io/github/license/amortized-ai/amortized?style=flat-square" alt="License"></a>
</p>

---

Take any task your AI agent handles with a frontier model and train a small, fast, cheap model that does it just as well.

```
Generate data  →  Train  →  Serve  →  Evaluate
   (asynth)       (TRL)    (vLLM)    (asynth)
```

## Get Started

```bash
pip install -e .
amortized config   # configure GPU backend
amortized up       # start server
```

## Run an Example

```bash
# generate → train → serve → evaluate
amortized submit sdg --recipe examples/ticket-classifier/synth --confirm
amortized submit training --recipe examples/ticket-classifier/train --data <id> --confirm
amortized submit serve --model Qwen/Qwen2.5-1.5B-Instruct --adapter <id> --confirm
amortized submit eval --recipe examples/ticket-classifier/eval --serve <id> --confirm
```

See [examples/](examples/) for 6 end-to-end projects: ticket classifier, intent router, entity extractor, summarizer, content moderator, and model distillation.

## License

[Apache 2.0](LICENSE)
