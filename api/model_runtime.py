"""
Chargement UNIQUE du modèle au démarrage de l'API.

Défaut : ONNX Runtime (étape 4 — latence).
Fallback : LightGBM + pandas si SCORING_RUNTIME=lgbm ou ONNX absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from api.config import (
    CAT_MAPS_PATH,
    MODEL_PATH,
    ONNX_PATH,
    SCORING_RUNTIME,
    get_threshold,
    load_cat_maps,
    load_schema,
)
from api.encode import encode_row


class Predictor(Protocol):
    feature_names: list[str]
    threshold: float
    backend: str

    def predict_proba(self, features: dict[str, Any]) -> float: ...

    def decide(self, proba: float) -> str: ...


@dataclass
class OnnxRuntime:
    session: Any
    feature_names: list[str]
    category_maps: dict[str, dict[str, int]]
    threshold: float
    backend: str = "onnx"

    def predict_proba(self, features: dict[str, Any]) -> float:
        arr = encode_row(features, self.feature_names, self.category_maps)
        _label, probs = self.session.run(None, {"input": arr})
        return float(probs[0, 1])

    def decide(self, proba: float) -> str:
        return "REFUS" if proba >= self.threshold else "ACCORD"


@dataclass
class LgbmRuntime:
    booster: Any
    feature_names: list[str]
    categorical_features: list[str]
    threshold: float
    backend: str = "lgbm"

    def predict_proba(self, features: dict[str, Any]) -> float:
        row = {name: features.get(name) for name in self.feature_names}
        frame = pd.DataFrame([row], columns=self.feature_names)
        cat_set = set(self.categorical_features)
        for col in self.feature_names:
            if col in cat_set:
                frame[col] = frame[col].astype("object").where(frame[col].notna(), None)
                frame[col] = frame[col].astype("category")
            else:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return float(self.booster.predict(frame)[0])

    def decide(self, proba: float) -> str:
        return "REFUS" if proba >= self.threshold else "ACCORD"


_runtime: Predictor | None = None


def get_runtime() -> Predictor:
    if _runtime is None:
        raise RuntimeError("Modèle non chargé : lifespan FastAPI non initialisé.")
    return _runtime


def _load_onnx() -> OnnxRuntime:
    import onnxruntime as ort

    schema = load_schema()
    cat_payload = load_cat_maps()
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    feature_names = list(cat_payload.get("feature_names") or schema["features"])
    return OnnxRuntime(
        session=session,
        feature_names=feature_names,
        category_maps=dict(cat_payload["category_maps"]),
        threshold=get_threshold(),
    )


def _load_lgbm() -> LgbmRuntime:
    import lightgbm as lgb

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle champion introuvable: {MODEL_PATH}. "
            "Lancer run_shap_lgbm / run_register_champion ou restaurer models/."
        )
    schema = load_schema()
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    feature_names = list(schema["features"])
    booster_names = list(booster.feature_name())
    if booster_names != feature_names:
        feature_names = booster_names
    return LgbmRuntime(
        booster=booster,
        feature_names=feature_names,
        categorical_features=list(schema.get("categorical_features", [])),
        threshold=get_threshold(),
    )


def load_runtime() -> Predictor:
    """Appelé une fois au startup."""
    global _runtime
    prefer = SCORING_RUNTIME
    if prefer not in {"onnx", "lgbm"}:
        prefer = "onnx"

    if prefer == "onnx":
        if ONNX_PATH.exists() and CAT_MAPS_PATH.exists():
            _runtime = _load_onnx()
            return _runtime
        # Fallback silencieux si artefacts ONNX absents (dev partiel).
        _runtime = _load_lgbm()
        return _runtime

    _runtime = _load_lgbm()
    return _runtime


def unload_runtime() -> None:
    global _runtime
    _runtime = None
