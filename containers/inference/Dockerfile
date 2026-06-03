FROM python:3.11-slim

COPY containers/shared/ /app/shared/
COPY containers/inference/runner.py /app/

WORKDIR /app

RUN pip install --no-cache-dir vllm

CMD ["python", "runner.py"]
