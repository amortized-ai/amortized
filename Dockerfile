FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml alembic.ini .
COPY src/ src/
COPY openrouter_costs.json .

RUN uv pip install --system --no-cache .

USER 1001

ENV AMORTIZED_DATABASE_URL=postgresql://amortized:amortized@localhost:5432/amortized \
    AMORTIZED_DATA_DIR=/data \
    AMORTIZED_RECIPES_DIR=/app

EXPOSE 8000
CMD ["uvicorn", "amortized.main:app", "--host", "0.0.0.0", "--port", "8000"]
