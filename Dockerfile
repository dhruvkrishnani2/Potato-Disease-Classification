# SpudGuard API — Render deployment image
# (Frontend is deployed separately on Vercel, so this image is backend-only.)
FROM python:3.12-slim

WORKDIR /app

# System deps needed to build a couple of the Python wheels (e.g. bcrypt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy backend code and the trained model
COPY api/ ./api/
COPY saved_models/ ./saved_models/

WORKDIR /app/api

# Render sets $PORT at runtime; main.py already reads it via os.environ
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/ping')" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
