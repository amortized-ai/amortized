FROM python:3.11-slim

COPY containers/shared/ /app/shared/
COPY containers/training/runner.py /app/

WORKDIR /app

RUN pip install --no-cache-dir training-hub[lora]

CMD ["python", "runner.py"]
