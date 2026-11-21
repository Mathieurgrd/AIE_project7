# Image API scoring crédit (partie 2) — deps runtime seulement.
# Build : docker build -t credit-scoring-api .
# Run  : docker run --rm -p 8000:8000 credit-scoring-api

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY models ./models
COPY reports/api_feature_schema.json ./reports/api_feature_schema.json
COPY reports/api_sample_client.json ./reports/api_sample_client.json

ENV PYTHONPATH=/app
EXPOSE 8000

# Modèle chargé au startup (lifespan FastAPI), pas à chaque requête.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
