# Model-liveness check service. A dedicated image with only what the checker needs —
# Python, FastAPI/Uvicorn, Pydantic, HTTPX, and the self-contained amortized.model_check
# package.
#
# Build from the repo root:
#   docker build -f containers/model-check/Dockerfile -t model-check .
FROM python:3.12-slim

RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/usr/local/bin/uv \
    uv pip install --system --no-cache \
      "fastapi>=0.115.0" \
      "uvicorn>=0.30.0" \
      "pydantic>=2.7.0" \
      "httpx>=0.27.0"

WORKDIR /app
COPY src/amortized/__init__.py amortized/__init__.py
COPY src/amortized/model_check/ amortized/model_check/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 1001
EXPOSE 7351
ENTRYPOINT ["python3", "-m", "amortized.model_check"]
CMD ["serve", "--port", "7351"]
