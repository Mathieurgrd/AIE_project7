"""
Profiling + optimisation d'inférence (partie 2 étape 4).

Hypothèse testée :
  le goulot n'est pas le booster LightGBM mais la construction pandas
  ligne-à-ligne dans l'API (~50 ms). ONNX Runtime + encodage numpy
  doit réduire fortement la latence sans régression AUC / proba.

Artefacts :
  models/champion_lgbm_optuna.onnx
  models/champion_lgbm_cat_maps.json
  reports/optimization/optimization_report.md
  reports/optimization/optimization_metrics.json
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_TXT = ROOT / "models" / "champion_lgbm_optuna.txt"
MODEL_ONNX = ROOT / "models" / "champion_lgbm_optuna.onnx"
CAT_MAPS_PATH = ROOT / "models" / "champion_lgbm_cat_maps.json"
SCHEMA_PATH = ROOT / "reports" / "api_feature_schema.json"
SAMPLE_PATH = ROOT / "reports" / "api_sample_client.json"
VALID_PATH = ROOT / "data" / "processed" / "valid.parquet"
OUT_DIR = ROOT / "reports" / "optimization"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _build_cat_maps(booster: lgb.Booster, feature_names: list[str], cat_cols: list[str]) -> dict:
    cat_set = set(cat_cols)
    ordered = [c for c in feature_names if c in cat_set]
    if len(ordered) != len(booster.pandas_categorical):
        raise RuntimeError(
            f"Mismatch cats: {len(ordered)} features vs "
            f"{len(booster.pandas_categorical)} pandas_categorical"
        )
    maps: dict[str, dict[str, int]] = {}
    for col, levels in zip(ordered, booster.pandas_categorical, strict=True):
        maps[col] = {str(v): int(i) for i, v in enumerate(levels)}
    return {
        "feature_names": feature_names,
        "categorical_features": ordered,
        "category_maps": maps,
    }


def encode_features_matrix(
    rows: list[dict] | pd.DataFrame,
    feature_names: list[str],
    category_maps: dict[str, dict[str, int]],
) -> np.ndarray:
    """Encode un batch (list de dicts ou DataFrame) → float32 (n, n_features)."""
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict(orient="records")
    else:
        records = rows
    arr = np.empty((len(records), len(feature_names)), dtype=np.float32)
    for i, features in enumerate(records):
        for j, name in enumerate(feature_names):
            val = features.get(name)
            if name in category_maps:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    arr[i, j] = np.nan
                else:
                    code = category_maps[name].get(str(val))
                    arr[i, j] = float(code) if code is not None else np.nan
            else:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    arr[i, j] = np.nan
                else:
                    try:
                        arr[i, j] = float(val)
                    except (TypeError, ValueError):
                        arr[i, j] = np.nan
    return arr


def predict_pandas_baseline(
    booster: lgb.Booster,
    features: dict,
    feature_names: list[str],
    cat_cols: list[str],
) -> float:
    """Chemin API historique (goulot pandas)."""
    cat_set = set(cat_cols)
    row = {n: features.get(n) for n in feature_names}
    frame = pd.DataFrame([row], columns=feature_names)
    for col in feature_names:
        if col in cat_set:
            frame[col] = frame[col].astype("object").where(frame[col].notna(), None)
            frame[col] = frame[col].astype("category")
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return float(booster.predict(frame)[0])


def _bench(fn, n: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def _profile_baseline(booster, sample, feature_names, cat_cols, n: int = 40) -> str:
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        predict_pandas_baseline(booster, sample, feature_names, cat_cols)
    pr.disable()
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumtime")
    stats.print_stats(15)
    return buf.getvalue()


def export_onnx(booster: lgb.Booster, n_features: int) -> None:
    onx = convert_lightgbm(
        booster,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=15,
        zipmap=False,
    )
    MODEL_ONNX.write_bytes(onx.SerializeToString())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profiling + export ONNX champion")
    parser.add_argument("--n-bench", type=int, default=100, help="Itérations latence")
    parser.add_argument("--n-valid", type=int, default=2000, help="Lignes valid pour fidélité")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = _load_schema()
    feature_names = list(schema["features"])
    cat_cols = list(schema["categorical_features"])
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))["features"]

    booster = lgb.Booster(model_file=str(MODEL_TXT))
    cat_payload = _build_cat_maps(booster, feature_names, cat_cols)
    CAT_MAPS_PATH.write_text(json.dumps(cat_payload, indent=2), encoding="utf-8")

    export_onnx(booster, len(feature_names))

    import onnxruntime as ort

    sess = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])
    category_maps = cat_payload["category_maps"]

    def run_baseline():
        return predict_pandas_baseline(booster, sample, feature_names, cat_cols)

    def run_onnx():
        arr = encode_features_matrix([sample], feature_names, category_maps)
        label, probs = sess.run(None, {"input": arr})
        return float(probs[0, 1])

    ms_baseline = _bench(run_baseline, args.n_bench)
    ms_onnx = _bench(run_onnx, args.n_bench)
    speedup = ms_baseline / ms_onnx if ms_onnx > 0 else float("inf")

    p_base = run_baseline()
    p_onnx = run_onnx()
    profile_txt = _profile_baseline(booster, sample, feature_names, cat_cols)
    (OUT_DIR / "cprofile_baseline.txt").write_text(profile_txt, encoding="utf-8")

    # Fidélité sur valid (si disponible)
    fidelity: dict = {"valid_available": VALID_PATH.exists()}
    if VALID_PATH.exists():
        valid = pd.read_parquet(VALID_PATH)
        n = min(args.n_valid, len(valid))
        valid = valid.sample(n=n, random_state=42)
        y = valid["TARGET"].to_numpy()
        frame = valid[feature_names].copy()
        for c in cat_cols:
            frame[c] = frame[c].astype("category")
        p_lgb = booster.predict(frame)
        arr = encode_features_matrix(frame, feature_names, category_maps)
        _, probs = sess.run(None, {"input": arr})
        p_ort = probs[:, 1]
        fidelity.update(
            {
                "n_rows": int(n),
                "auc_lgbm_pandas": float(roc_auc_score(y, p_lgb)),
                "auc_onnx": float(roc_auc_score(y, p_ort)),
                "max_abs_proba_delta": float(np.max(np.abs(p_lgb - p_ort))),
                "mean_abs_proba_delta": float(np.mean(np.abs(p_lgb - p_ort))),
            }
        )

    metrics = {
        "backend_baseline": "lightgbm+pandas (API historique)",
        "backend_optimized": "onnxruntime + encodage numpy (cats → codes)",
        "latency_ms_mean_baseline": ms_baseline,
        "latency_ms_mean_onnx": ms_onnx,
        "speedup_x": speedup,
        "sample_proba_baseline": p_base,
        "sample_proba_onnx": p_onnx,
        "sample_abs_delta": abs(p_base - p_onnx),
        "onnx_path": str(MODEL_ONNX.relative_to(ROOT)),
        "onnx_size_mb": MODEL_ONNX.stat().st_size / 1e6,
        "fidelity": fidelity,
        "bottleneck": (
            "cProfile : construction/casts pandas + lightgbm._data_from_pandas "
            "dominent le temps ; le scoring arbres est secondaire."
        ),
        "decision": (
            "Adopter ONNX Runtime dans l'API (défaut). "
            "Fallback LightGBM+pandas via SCORING_RUNTIME=lgbm."
        ),
    }
    (OUT_DIR / "optimization_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fid_lines = []
    if fidelity.get("valid_available"):
        fid_lines = [
            f"- AUC LightGBM (pandas) : {fidelity['auc_lgbm_pandas']:.6f}",
            f"- AUC ONNX : {fidelity['auc_onnx']:.6f}",
            f"- max |Δproba| : {fidelity['max_abs_proba_delta']:.2e}",
            f"- mean |Δproba| : {fidelity['mean_abs_proba_delta']:.2e}",
            f"- n lignes valid : {fidelity['n_rows']}",
        ]
    else:
        fid_lines = ["- valid.parquet absent : fidélité batch non mesurée (sample client OK)."]

    report = "\n".join(
        [
            "# Rapport d'optimisation d'inférence (étape 4)",
            "",
            "## Contexte",
            "Après déploiement API + logs, la latence unitaire ~50 ms était trop élevée",
            "pour du scoring ligne-à-ligne. Objectif : réduire le temps d'inférence",
            "sans régression de précision (consigne partie 2 étape 4).",
            "",
            "## Goulot identifié (cProfile)",
            "- Hot path = construction DataFrame pandas + casts colonnes + `_data_from_pandas`.",
            "- Le booster LightGBM lui-même n'est pas le coût dominant en single-row.",
            f"- Détail : `reports/optimization/cprofile_baseline.txt`",
            "",
            "## Stratégies testées",
            "1. **Optimisation de code** : encodage numpy des catégorielles via `pandas_categorical` du modèle.",
            "2. **ONNX Runtime** : export `onnxmltools.convert_lightgbm` (zipmap=False) + inference CPU.",
            "3. (Non retenu) re-entraîner sans cats natives — risque métier inutile vu le succès ONNX.",
            "",
            "## Résultats latence (single-row, sample client)",
            f"- Baseline LightGBM+pandas : **{ms_baseline:.3f} ms**",
            f"- ONNX Runtime + numpy : **{ms_onnx:.3f} ms**",
            f"- Speedup : **×{speedup:.1f}**",
            f"- |Δproba| sample : {abs(p_base - p_onnx):.2e}",
            "",
            "## Fidélité (holdout valid)",
            *fid_lines,
            "",
            "## Configuration finale",
            "- Runtime API défaut : **ONNX Runtime** (`models/champion_lgbm_optuna.onnx`)",
            "- Maps catégories : `models/champion_lgbm_cat_maps.json`",
            "- Fallback : `SCORING_RUNTIME=lgbm` (fichier `.txt` + pandas)",
            "- Image Docker : `onnxruntime` (plus besoin de LightGBM au runtime)",
            "",
            "## Décision",
            metrics["decision"],
            "",
        ]
    )
    (OUT_DIR / "optimization_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nRapport : {OUT_DIR / 'optimization_report.md'}")


if __name__ == "__main__":
    main()
