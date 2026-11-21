"""Validation métier des inputs API (champs manquants, types, plages)."""

from __future__ import annotations

from typing import Any

from api.config import load_schema


class InputValidationError(Exception):
    """Erreur de validation renvoyée en HTTP 422 avec détail structuré."""

    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")


def validate_features(features: dict[str, Any]) -> dict[str, Any]:
    schema = load_schema()
    required = list(schema["features"])
    cats = set(schema.get("categorical_features", []))
    nums = set(schema.get("numeric_features", []))
    errors: list[dict[str, Any]] = []

    missing = [name for name in required if name not in features]
    if missing:
        # Trop de colonnes pour tout lister si payload vide : on borne l'affichage.
        preview = missing[:20]
        errors.append(
            {
                "code": "missing_features",
                "message": f"{len(missing)} feature(s) manquante(s)",
                "missing_preview": preview,
                "missing_count": len(missing),
            }
        )

    # Contrôles type / plage uniquement sur les clés présentes.
    for name, value in features.items():
        if name not in required:
            # Extra keys : on ignore (forward-compat) plutôt que d'échouer.
            continue
        if value is None:
            # NA autorisé (LightGBM gère les manquants).
            continue

        if name in nums:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    {
                        "code": "invalid_type",
                        "feature": name,
                        "expected": "number",
                        "got": type(value).__name__,
                        "value": value,
                    }
                )
                continue
            # Plages « sens métier » (consigne : âge négatif, revenu invalide, etc.).
            if name in {"AGE_YEARS", "CNT_CHILDREN", "AMT_CREDIT", "AMT_ANNUITY"} and float(value) < 0:
                errors.append(
                    {
                        "code": "out_of_range",
                        "feature": name,
                        "message": f"{name} ne peut pas être négatif",
                        "value": value,
                    }
                )
            if name == "AMT_INCOME_TOTAL" and float(value) <= 0:
                errors.append(
                    {
                        "code": "out_of_range",
                        "feature": name,
                        "message": "AMT_INCOME_TOTAL doit être > 0",
                        "value": value,
                    }
                )
            if name == "AGE_YEARS" and float(value) > 120:
                errors.append(
                    {
                        "code": "out_of_range",
                        "feature": name,
                        "message": "AGE_YEARS > 120 improbable",
                        "value": value,
                    }
                )

        if name in cats and not isinstance(value, str):
            errors.append(
                {
                    "code": "invalid_type",
                    "feature": name,
                    "expected": "string",
                    "got": type(value).__name__,
                    "value": value,
                }
            )

    if errors:
        raise InputValidationError(errors)

    # Ne garder que les features attendues (ordre d'entraînement côté runtime).
    return {name: features.get(name) for name in required}
