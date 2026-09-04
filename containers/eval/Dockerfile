# Eval container — compare model endpoints on an eval dataset (CPU-only)
FROM docker.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "httpx>=0.27" "pyarrow>=16"

COPY run_eval.py /app/run_eval.py

WORKDIR /amortized/work

ENTRYPOINT ["python3", "/app/run_eval.py"]
