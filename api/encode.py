"""Encodage features API → matrice float32 pour ONNX / LightGBM numpy."""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_row(
    features: dict[str, Any],
    feature_names: list[str],
    category_maps: dict[str, dict[str, int]],
) -> np.ndarray:
    """Une ligne → shape (1, n_features) float32."""
    row = np.empty((1, len(feature_names)), dtype=np.float32)
    for j, name in enumerate(feature_names):
        val = features.get(name)
        if name in category_maps:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row[0, j] = np.nan
            else:
                code = category_maps[name].get(str(val))
                row[0, j] = float(code) if code is not None else np.nan
        else:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row[0, j] = np.nan
            else:
                try:
                    row[0, j] = float(val)
                except (TypeError, ValueError):
                    row[0, j] = np.nan
    return row
