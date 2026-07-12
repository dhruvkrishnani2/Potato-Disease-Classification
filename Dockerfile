# SpudGuard API — Render deployment image
# (Frontend is deployed separately on Vercel, so this image is backend-only.)
# NOTE: api/ is imported as a Python package (e.g. `from api.config import ...`),
# so the app must run from the repo root, not from inside api/.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=model_v1.keras
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
