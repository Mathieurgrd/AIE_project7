# Image API scoring crédit (partie 2) — deps runtime seulement.
# Build : docker build -t credit-scoring-api .
# Run  : docker run --rm -p 8000:8000 credit-scoring-api
# Runtime défaut : ONNX (étape 4). Fallback local : SCORING_RUNTIME=lgbm

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY models/champion_lgbm_optuna.onnx ./models/champion_lgbm_optuna.onnx
COPY models/champion_lgbm_cat_maps.json ./models/champion_lgbm_cat_maps.json
COPY reports/api_feature_schema.json ./reports/api_feature_schema.json
COPY reports/api_sample_client.json ./reports/api_sample_client.json

ENV PYTHONPATH=/app
ENV SCORING_RUNTIME=onnx
EXPOSE 8000

# Modèle chargé au startup (lifespan FastAPI), pas à chaque requête.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
