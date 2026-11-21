"""Chemins et constantes de l'API (seuil métier inclus)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "champion_lgbm_optuna.txt"
SCHEMA_PATH = ROOT / "reports" / "api_feature_schema.json"
LOG_DIR = ROOT / "logs"
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"

# Fallback si le schéma JSON est incomplet (ne devrait pas arriver en prod locale).
DEFAULT_THRESHOLD = 0.52


@lru_cache(maxsize=1)
def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schéma API manquant: {SCHEMA_PATH}. "
            "Générer via les colonnes fe_v1 (voir docs/Decision.md D-018)."
        )
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def get_threshold() -> float:
    schema = load_schema()
    return float(schema.get("threshold_opt_valid", DEFAULT_THRESHOLD))
