"""
Comparaison multi-modèles avec StratifiedKFold + seuil métier → MLflow.

Protocole (anti-leakage) :
  1) CV sur train.parquet → probabilités OOF (chaque ligne prédite hors fold)
  2) Seuil* = argmin coût 10*FN+FP sur ces OOF
  3) Refit sur tout le train → métriques holdout valid @0.5 et @seuil*
  4) Un run MLflow par famille (LR, RF, LGBM, XGB)

Usage :
  uv run python -m src.run_cv_baselines
  uv run python -m src.run_cv_baselines --models lightgbm xgboost
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.modeling import (
    FN_COST_DEFAULT,
    FP_COST_DEFAULT,
    build_estimators,
    eval_at_threshold,
    optimize_threshold,
    prepare_tree_frame,
)

# MLflow/skops exige de whitelister certains types pour log_model sklearn.
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.tree._tree.Tree"]

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
N_SPLITS = 5
RANDOM_STATE = 42
FN_COST = FN_COST_DEFAULT
FP_COST = FP_COST_DEFAULT


# ---------------------------------------------------------------------------
# Helpers CV / fit / predict
# ---------------------------------------------------------------------------

def _oof_predict(
    spec: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    cv: StratifiedKFold,
) -> tuple[np.ndarray, list[float]]:
    """Entraîne K fois : remplit oof[i] = proba quand i est dans le fold valid."""
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    needs_pre = spec["needs_sklearn_preprocess"]
    fit_kwargs = dict(spec.get("fit_kwargs") or {})

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        est = clone(spec["estimator"])  # nouvel estimateur vierge à chaque fold
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if not needs_pre:
            X_tr = prepare_tree_frame(X_tr, cat_cols)
            X_va = prepare_tree_frame(X_va, cat_cols)
            # Même vocabulaire de catégories train→valid (surtout XGBoost).
            for c in cat_cols:
                X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

        kw = dict(fit_kwargs)
        if spec["family"] == "xgboost":
            # Pondération déséquilibre calculée SUR le fold train seulement.
            n_pos = float((y_tr == 1).sum())
            n_neg = float((y_tr == 0).sum())
            est.set_params(scale_pos_weight=n_neg / max(n_pos, 1.0))

        if spec["family"] == "lightgbm":
            est.fit(X_tr, y_tr, **kw)
        else:
            est.fit(X_tr, y_tr)

        proba = est.predict_proba(X_va)[:, 1]  # proba de TARGET=1 (défaut)
        oof[va_idx] = proba
        fold_aucs.append(float(roc_auc_score(y_va, proba)))
        print(f"  fold {fold}/{cv.n_splits} AUC={fold_aucs[-1]:.4f}", flush=True)

    return oof, fold_aucs


def _fit_full(spec: dict, X: pd.DataFrame, y: pd.Series, cat_cols: list[str]):
    """Refit sur 100 % du train (pour l'éval holdout valid)."""
    est = clone(spec["estimator"])
    fit_kwargs = dict(spec.get("fit_kwargs") or {})
    X_fit = X if spec["needs_sklearn_preprocess"] else prepare_tree_frame(X, cat_cols)

    if spec["family"] == "xgboost":
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        est.set_params(scale_pos_weight=n_neg / max(n_pos, 1.0))

    if spec["family"] == "lightgbm":
        est.fit(X_fit, y, **fit_kwargs)
    else:
        est.fit(X_fit, y)
    return est, X_fit


def _predict_proba(spec: dict, est, X: pd.DataFrame, cat_cols: list[str], ref_cats: pd.DataFrame):
    """Proba sur un jeu externe (valid), en alignant les catégories si besoin."""
    if spec["needs_sklearn_preprocess"]:
        return est.predict_proba(X)[:, 1]
    X_prep = prepare_tree_frame(X, cat_cols)
    for c in cat_cols:
        X_prep[c] = pd.Categorical(X_prep[c], categories=ref_cats[c].cat.categories)
    return est.predict_proba(X_prep)[:, 1]


def _log_threshold_curve(curve: pd.DataFrame, run_name: str) -> None:
    """CSV + PNG coût vs seuil → artefacts MLflow."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "threshold_cost_curve.csv"
        curve.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path), artifact_path="threshold")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(curve["threshold"], curve["business_cost"], color="#1f4e79")
        best = curve.loc[curve["business_cost"].idxmin()]
        ax.axvline(best["threshold"], color="#c44e52", linestyle="--", label=f"seuil*={best['threshold']:.2f}")
        ax.set_xlabel("Seuil")
        ax.set_ylabel(f"Coût métier ({FN_COST}*FN + {FP_COST}*FP)")
        ax.set_title(f"Coût vs seuil — {run_name}")
        ax.legend()
        fig.tight_layout()
        png_path = tmp_path / "threshold_cost_curve.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(png_path), artifact_path="threshold")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV + seuil metier -> MLflow")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Sous-ensemble: logreg random_forest lightgbm xgboost",
    )
    return parser.parse_args()


def _log_fitted_model(spec: dict, est) -> None:
    """Log le modèle dans MLflow ; si skops bloque, on garde quand même les métriques."""
    try:
        if spec["family"] == "lightgbm":
            mlflow.lightgbm.log_model(est, name="model")
        elif spec["family"] == "xgboost":
            mlflow.xgboost.log_model(est, name="model")
        else:
            mlflow.sklearn.log_model(
                est,
                name="model",
                skops_trusted_types=SKOPS_TRUSTED,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] log_model ignore: {exc}", flush=True)
        mlflow.set_tag("model_logged", "false")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    print("Chargement processed...", flush=True)
    meta = json.loads((PROC / "meta.json").read_text(encoding="utf-8"))
    train = pd.read_parquet(PROC / "train.parquet")
    valid = pd.read_parquet(PROC / "valid.parquet")

    y_train = train["TARGET"].astype(int)
    y_valid = valid["TARGET"].astype(int)
    X_train = train.drop(columns=["TARGET", "SK_ID_CURR"])
    X_valid = valid.drop(columns=["TARGET", "SK_ID_CURR"])

    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Même URI SQLite que l'UI MLflow
    tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-scoring-fe_v1")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    specs = build_estimators(num_cols, cat_cols, random_state=RANDOM_STATE)
    if args.models:
        unknown = set(args.models) - set(specs)
        if unknown:
            raise SystemExit(f"Modeles inconnus: {sorted(unknown)}")
        specs = {k: specs[k] for k in args.models}

    print(f"fe_version={meta['fe_version']} | train={len(train)} valid={len(valid)}", flush=True)
    print(f"features={X_train.shape[1]} num={len(num_cols)} cat={len(cat_cols)}", flush=True)
    print(f"CV=StratifiedKFold(n_splits={N_SPLITS}) | cout FN:FP={FN_COST}:{FP_COST}", flush=True)
    print(f"tracking={tracking_uri}", flush=True)
    print(f"models={list(specs)}", flush=True)

    summary_rows: list[dict] = []

    for name, spec in specs.items():
        run_name = f"{name}_cv{N_SPLITS}_seuil_opt_fe_v1"
        print(f"\n=== {run_name} ===", flush=True)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "fe_version": meta["fe_version"],
                    "model_family": spec["family"],
                    "stage": "cv_threshold",
                    "cv_strategy": "StratifiedKFold",
                }
            )
            mlflow.log_params(
                {
                    **spec["params"],
                    "n_splits": N_SPLITS,
                    "random_state": RANDOM_STATE,
                    "fn_cost": FN_COST,
                    "fp_cost": FP_COST,
                    "threshold_search": "oof_argmin_business_cost",
                }
            )

            # --- Étape CV ---
            oof, fold_aucs = _oof_predict(spec, X_train, y_train, cat_cols, cv)
            auc_oof = float(roc_auc_score(y_train, oof))
            best_thr, curve = optimize_threshold(
                y_train, oof, fn_cost=FN_COST, fp_cost=FP_COST
            )
            metrics_oof_05 = eval_at_threshold(
                y_train, oof, threshold=0.5, fn_cost=FN_COST, fp_cost=FP_COST
            )
            metrics_oof_star = eval_at_threshold(
                y_train, oof, threshold=best_thr, fn_cost=FN_COST, fp_cost=FP_COST
            )

            # --- Étape holdout valid ---
            est, X_train_fit = _fit_full(spec, X_train, y_train, cat_cols)
            ref = X_train_fit if not spec["needs_sklearn_preprocess"] else X_train
            proba_valid = _predict_proba(spec, est, X_valid, cat_cols, ref)
            metrics_val_05 = eval_at_threshold(
                y_valid, proba_valid, threshold=0.5, fn_cost=FN_COST, fp_cost=FP_COST
            )
            metrics_val_star = eval_at_threshold(
                y_valid, proba_valid, threshold=best_thr, fn_cost=FN_COST, fp_cost=FP_COST
            )

            mlflow.log_metrics(
                {
                    "cv_auc_mean": float(np.mean(fold_aucs)),
                    "cv_auc_std": float(np.std(fold_aucs)),
                    "oof_auc": auc_oof,
                    "oof_business_cost_at_0.5": float(metrics_oof_05["business_cost"]),
                    "oof_business_cost_at_opt": float(metrics_oof_star["business_cost"]),
                    "threshold_opt": best_thr,
                    "valid_auc": float(metrics_val_star["auc"]),
                    "valid_accuracy_at_0.5": float(metrics_val_05["accuracy"]),
                    "valid_accuracy_at_opt": float(metrics_val_star["accuracy"]),
                    "valid_business_cost_at_0.5": float(metrics_val_05["business_cost"]),
                    "valid_business_cost_at_opt": float(metrics_val_star["business_cost"]),
                    "valid_fn_at_opt": float(metrics_val_star["fn"]),
                    "valid_fp_at_opt": float(metrics_val_star["fp"]),
                    "valid_recall_at_opt": float(metrics_val_star["recall"]),
                    "valid_f1_at_opt": float(metrics_val_star["f1"]),
                }
            )
            _log_threshold_curve(curve, run_name)
            _log_fitted_model(spec, est)

            summary_rows.append(
                {
                    "model": name,
                    "cv_auc_mean": float(np.mean(fold_aucs)),
                    "cv_auc_std": float(np.std(fold_aucs)),
                    "threshold_opt": best_thr,
                    "oof_cost_0.5": float(metrics_oof_05["business_cost"]),
                    "oof_cost_opt": float(metrics_oof_star["business_cost"]),
                    "valid_auc": float(metrics_val_star["auc"]),
                    "valid_cost_0.5": float(metrics_val_05["business_cost"]),
                    "valid_cost_opt": float(metrics_val_star["business_cost"]),
                }
            )
            print(
                f"  OOF AUC={auc_oof:.4f} | seuil*={best_thr:.2f} | "
                f"cost OOF 0.5->opt: {metrics_oof_05['business_cost']:.0f}->{metrics_oof_star['business_cost']:.0f}",
                flush=True,
            )
            print(
                f"  VALID AUC={metrics_val_star['auc']:.4f} | "
                f"cost 0.5->opt: {metrics_val_05['business_cost']:.0f}->{metrics_val_star['business_cost']:.0f}",
                flush=True,
            )

    # Tableau local pour le notebook / la soutenance
    summary = pd.DataFrame(summary_rows).sort_values("valid_cost_opt")
    out = ROOT / "reports" / "cv_threshold_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print("\n=== Résumé (trié par valid_cost_opt) ===")
    print(summary.to_string(index=False))
    print(f"\nSauvé: {out}")


if __name__ == "__main__":
    main()
