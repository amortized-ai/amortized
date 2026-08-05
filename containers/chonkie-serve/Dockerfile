FROM python:3.12-slim

RUN pip install --no-cache-dir "chonkie[all]>=1.7.0" greenlet aiosqlite sqlalchemy

EXPOSE 8900

CMD ["chonkie", "serve", "--port", "8900"]
