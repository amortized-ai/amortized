FROM python:3.12-slim
WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -e .

USER 1001
ENV AMORTIZED_DB_PATH=/data/amortized.db \
    AMORTIZED_DATA_DIR=/data

EXPOSE 8000
CMD ["uvicorn", "amortized.main:app", "--host", "0.0.0.0", "--port", "8000"]
