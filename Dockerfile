# Image API scoring crédit (partie 2).
# Build : docker build -t credit-scoring-api .
# Run  : docker run --rm -p 8000:8000 credit-scoring-api

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY api ./api
COPY models ./models
COPY reports ./reports

# Dépendances lockées ; PYTHONPATH pour importer api/ et src/
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

EXPOSE 8000

# Modèle chargé au startup (lifespan FastAPI), pas à chaque requête.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
