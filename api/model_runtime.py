"""
Chargement UNIQUE du booster LightGBM au démarrage de l'API.

Pourquoi : la consigne impose de ne pas recharger le modèle à chaque requête
(latence + mémoire). On expose un singleton rempli dans le lifespan FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from api.config import MODEL_PATH, get_threshold, load_schema


@dataclass
class ChampionRuntime:
    booster: lgb.Booster
    feature_names: list[str]
    categorical_features: list[str]
    threshold: float

    def predict_proba(self, features: dict[str, Any]) -> float:
        """Construit une ligne alignée sur le schéma d'entraînement, puis score."""
        row = {name: features.get(name) for name in self.feature_names}
        frame = pd.DataFrame([row], columns=self.feature_names)
        cat_set = set(self.categorical_features)
        for col in self.feature_names:
            if col in cat_set:
                # None → NA catégoriel ; cast string sinon.
                frame[col] = frame[col].astype("object").where(frame[col].notna(), None)
                frame[col] = frame[col].astype("category")
            else:
                # JSON null arrive en object ; forcer float (NA OK pour LightGBM).
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        proba = float(self.booster.predict(frame)[0])
        return proba

    def decide(self, proba: float) -> str:
        return "REFUS" if proba >= self.threshold else "ACCORD"


_runtime: ChampionRuntime | None = None


def get_runtime() -> ChampionRuntime:
    if _runtime is None:
        raise RuntimeError("Modèle non chargé : lifespan FastAPI non initialisé.")
    return _runtime


def load_runtime() -> ChampionRuntime:
    """Appelé une fois au startup."""
    global _runtime
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle champion introuvable: {MODEL_PATH}. "
            "Lancer run_shap_lgbm / run_register_champion ou restaurer models/."
        )
    schema = load_schema()
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    feature_names = list(schema["features"])
    # Sécurité : aligner sur les noms du booster si divergence.
    booster_names = list(booster.feature_name())
    if booster_names != feature_names:
        feature_names = booster_names
    _runtime = ChampionRuntime(
        booster=booster,
        feature_names=feature_names,
        categorical_features=list(schema.get("categorical_features", [])),
        threshold=get_threshold(),
    )
    return _runtime


def unload_runtime() -> None:
    global _runtime
    _runtime = None
