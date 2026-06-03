FROM python:3.11-slim

COPY containers/shared/ /app/shared/
COPY containers/eval/runner.py /app/

WORKDIR /app

RUN pip install --no-cache-dir litellm

CMD ["python", "runner.py"]
