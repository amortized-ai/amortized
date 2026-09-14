# Eval container — evaluates a model on a dataset, serving the model itself.
#
# Based on the training image (vLLM + model tooling) so the eval pod can
# bring up the model on a pinned GPU, wait for it to be healthy, run the
# eval against localhost, and exit — the serve lifetime equals the eval
# lifetime. External-endpoint evals run the same image CPU-only.
# httpx and pyarrow are already in the training base.
FROM ghcr.io/amortized-ai/training:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY run_eval.py /app/run_eval.py

WORKDIR /amortized/work

ENTRYPOINT ["python3", "/app/run_eval.py"]
