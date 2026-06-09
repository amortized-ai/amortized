FROM docker.io/python:3.11-slim
LABEL org.opencontainers.image.source=https://github.com/amortized-ai/amortized

COPY containers/shared/ /app/shared/
COPY containers/eval/runner.py /app/

WORKDIR /app

RUN pip install --no-cache-dir litellm

CMD ["python", "runner.py"]
