FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY server.py .

RUN mkdir -p /app/workspace /data && chown 1001:0 /app/workspace /data

USER 1001

ENV CLAUDE_CONFIG_DIR=/data \
    PYTHONUNBUFFERED=1

EXPOSE 4096

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4096"]
