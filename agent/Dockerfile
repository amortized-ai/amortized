FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

RUN mkdir -p /app/workspace /data && chown 1001:0 /app/workspace /data

USER 1001

ENV CLAUDE_CONFIG_DIR=/data \
    PYTHONUNBUFFERED=1

EXPOSE 4096

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4096"]
