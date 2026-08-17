FROM python:3.12-slim
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN sh /tmp/uv-install.sh && mv /root/.local/bin/uv /usr/local/bin/uv && rm /tmp/uv-install.sh
WORKDIR /app

COPY pyproject.toml alembic.ini .
COPY src/ src/
COPY templates/ templates/
COPY openrouter_costs.json .

RUN uv pip install --system --no-cache .

USER 1001

ENV AMORTIZED_DATABASE_URL=postgresql://amortized:amortized@localhost:5432/amortized \
    AMORTIZED_DATA_DIR=/data \
    AMORTIZED_RECIPES_DIR=/app

EXPOSE 8000
CMD ["uvicorn", "amortized.main:app", "--host", "0.0.0.0", "--port", "8000"]
