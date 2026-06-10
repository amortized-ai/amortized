# amortized

*Build task models that replace frontier API calls*

---

Every AI agent has tasks that don't need a frontier model. Classification, extraction, routing, summarization — these are specific, repeatable, and learnable. A small fine-tuned model can do them faster, cheaper, and more reliably than a general-purpose API.

**Amortized builds these task models.** You describe the task, it generates training data from a teacher model, fine-tunes a small student model, and evaluates whether the student matches the teacher. The result: a model you own that runs on your infrastructure, costs a fraction per inference, and doesn't break when the API provider changes.

The name comes from finance — amortization spreads a large upfront cost across many future uses. Here, the "cost" is the frontier model's capability, and the "uses" are every future inference by the cheaper task model.

## Get Started

```bash
pip install -e .
amortized config   # configure GPU backend
amortized up       # start server
```

## Run an Example

```bash
amortized submit sdg --recipe examples/ticket-classifier/synth --confirm
amortized submit training --recipe examples/ticket-classifier/train --data <id> --confirm
amortized submit serve --model Qwen/Qwen2.5-1.5B-Instruct --adapter <id> --confirm
amortized submit eval --recipe examples/ticket-classifier/eval --serve <id> --confirm
```

See [examples/](examples/) for end-to-end projects: ticket classifier, intent router, entity extractor, summarizer, content moderator, and model distillation.

## License

[Apache 2.0](LICENSE)
