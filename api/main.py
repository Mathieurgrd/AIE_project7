"""
API FastAPI — scoring crédit « Prêt à dépenser ».

Endpoints :
  GET  /health
  GET  /model/info
  POST /predict   body: {"features": {...}} → proba + décision (seuil*)

Lancer :
  uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.config import MODEL_PATH, SCHEMA_PATH, get_threshold, load_schema
from api.logging_utils import append_prediction_log
from api.model_runtime import get_runtime, load_runtime, unload_runtime
from api.validation import InputValidationError, validate_features


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Dictionnaire feature → valeur (schéma fe_v1, 218 colonnes).",
    )


class PredictResponse(BaseModel):
    proba_default: float
    threshold: float
    decision: str
    latency_ms: float


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Charge le booster UNE fois — exigence consigne partie 2.
    load_runtime()
    yield
    unload_runtime()


app = FastAPI(
    title="Prêt à dépenser — Credit Scoring API",
    description=(
        "API de prédiction de risque de défaut (LightGBM Optuna fe_v1). "
        "Le seuil métier (0.52) convertit la proba en ACCORD / REFUS."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(InputValidationError)
async def input_validation_handler(_request: Request, exc: InputValidationError):
    append_prediction_log(
        {
            "endpoint": "/predict",
            "status": "validation_error",
            "http_status": 422,
            "error_codes": [e.get("code") for e in exc.errors if isinstance(e, dict)],
            "n_errors": len(exc.errors),
        }
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors})


@app.exception_handler(RequestValidationError)
async def pydantic_validation_handler(_request: Request, exc: RequestValidationError):
    append_prediction_log(
        {
            "endpoint": "/predict",
            "status": "validation_error",
            "http_status": 422,
            "error": "request_validation_error",
        }
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    schema = load_schema()
    runtime = get_runtime()
    return {
        "model_path": str(MODEL_PATH),
        "schema_path": str(SCHEMA_PATH),
        "fe_version": schema.get("fe_version"),
        "n_features": len(runtime.feature_names),
        "threshold_opt_valid": runtime.threshold,
        "decision_rule": f"proba >= {runtime.threshold} → REFUS else ACCORD",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    try:
        features = validate_features(payload.features)
        runtime = get_runtime()
        proba = runtime.predict_proba(features)
        decision = runtime.decide(proba)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        append_prediction_log(
            {
                "endpoint": "/predict",
                "status": "ok",
                "http_status": 200,
                "n_features_received": len(payload.features),
                # Inputs/outputs complets pour drift (consigne étape 3).
                "features": features,
                "proba_default": proba,
                "threshold": runtime.threshold,
                "decision": decision,
                "latency_ms": latency_ms,
            }
        )
        return PredictResponse(
            proba_default=proba,
            threshold=runtime.threshold,
            decision=decision,
            latency_ms=round(latency_ms, 3),
        )
    except InputValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 — log + 500 contrôlé
        latency_ms = (time.perf_counter() - t0) * 1000.0
        append_prediction_log(
            {
                "endpoint": "/predict",
                "status": "error",
                "error": str(exc),
                "latency_ms": latency_ms,
            }
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
