FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY examples/ examples/
COPY templates/ templates/

RUN pip install --no-cache-dir .

USER 1001

ENV AMORTIZED_DB_PATH=/data/amortized.db \
    AMORTIZED_DATA_DIR=/data \
    AMORTIZED_RECIPES_DIR=/app

EXPOSE 8000
CMD ["uvicorn", "amortized.main:app", "--host", "0.0.0.0", "--port", "8000"]
