FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# predictable.db ships inside the image: the site only reads it, and data is
# ingested locally, so each deploy carries the latest snapshot.
COPY . .

EXPOSE 8000
# One worker with threads: each worker holds the model's caches in memory, and
# the container is capped at 1 GB.
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "8", "--timeout", "120", "--bind", "0.0.0.0:8000"]
