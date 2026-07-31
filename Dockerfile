FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY templates/ templates/
COPY openrouter_costs.json .

RUN uv pip install --system --no-cache .

USER 1001

ENV AMORTIZED_DB_PATH=/data/amortized.db \
    AMORTIZED_DATA_DIR=/data \
    AMORTIZED_RECIPES_DIR=/app

EXPOSE 8000
CMD ["uvicorn", "amortized.main:app", "--host", "0.0.0.0", "--port", "8000"]
