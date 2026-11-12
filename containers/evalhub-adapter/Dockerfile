FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp/amortized \
    HF_HOME=/tmp/amortized/huggingface PYTHONPATH=/app EVALHUB_MODE=k8s
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    'eval-hub-sdk[adapter]==1.0.7' 'httpx>=0.27,<1' \
    'pyarrow>=18,<25' 'sentence-transformers>=5,<6'
COPY src/amortized/__init__.py /app/amortized/__init__.py
COPY src/amortized/evaluation /app/amortized/evaluation
COPY src/amortized/jobs/assets/classify_eval.py /app/amortized/jobs/assets/classify_eval.py
RUN mkdir -p /tmp/amortized && chmod 1777 /tmp/amortized
WORKDIR /tmp/amortized
# Hub's init container (uid 1000) downloads test data into folders it creates as
# 0750. Kubernetes ignores a group in USER, so run as that uid to read them.
USER 1000
ENTRYPOINT ["python", "-m", "amortized.evaluation.adapter"]
