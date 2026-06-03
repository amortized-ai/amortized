FROM python:3.11-slim

COPY containers/shared/ /app/shared/
COPY containers/synth/runner.py /app/

WORKDIR /app

RUN pip install --no-cache-dir sdg-hub datasets

CMD ["python", "runner.py"]
