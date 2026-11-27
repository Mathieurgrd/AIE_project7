"""
Optimise le seuil métier sur le holdout valid (LightGBM baseline CV).

Rôle pédagogique : montrer que pour un modèle donné, on balaye les seuils
et on choisit celui qui minimise 10*FN+FP (ici souvent ~0.5).

Note : le champion Optuna a son propre seuil* dans run_optuna_lgbm
       (reports/optuna_lgbm_best.json → threshold_opt_valid).

Usage : uv run python -m src.run_threshold_on_valid
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier

from src.modeling import eval_at_threshold, optimize_threshold, prepare_tree_frame

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> None:
    meta = json.loads((PROC / "meta.json").read_text(encoding="utf-8"))
    train = pd.read_parquet(PROC / "train.parquet")
    valid = pd.read_parquet(PROC / "valid.parquet")
    y_tr = train["TARGET"].astype(int)
    y_va = valid["TARGET"].astype(int)
    X_tr = train.drop(columns=["TARGET", "SK_ID_CURR"])
    X_va = valid.drop(columns=["TARGET", "SK_ID_CURR"])
    cat = X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Même préparation que les autres scripts arbres
    X_tr = prepare_tree_frame(X_tr, cat)
    X_va = prepare_tree_frame(X_va, cat)
    for c in cat:
        X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

    # Hyperparams = baseline CV (pas Optuna) — volontaire pour isoler l'effet seuil
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb.fit(X_tr, y_tr, categorical_feature=cat)
    proba = lgb.predict_proba(X_va)[:, 1]

    # Balayage → seuil* + comparaison vs 0.5
    thr, curve = optimize_threshold(y_va, proba)
    m05 = eval_at_threshold(y_va, proba, 0.5)
    mstar = eval_at_threshold(y_va, proba, thr)

    print(f"valid_opt_threshold={thr}", flush=True)
    print(
        f"cost_0.5={m05['business_cost']:.0f} (fn={m05['fn']} fp={m05['fp']})",
        flush=True,
    )
    print(
        f"cost_opt={mstar['business_cost']:.0f} (fn={mstar['fn']} fp={mstar['fp']})",
        flush=True,
    )
    print(curve.nsmallest(5, "business_cost")[["threshold", "business_cost", "fn", "fp"]], flush=True)

    out = ROOT / "reports" / "lgbm_threshold_curve_valid.csv"
    curve.to_csv(out, index=False)

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("credit-scoring-fe_v1")
    with mlflow.start_run(run_name="lightgbm_seuil_valid_opt_fe_v1"):
        mlflow.set_tags(
            {
                "fe_version": meta["fe_version"],
                "model_family": "lightgbm",
                "stage": "threshold_on_valid",
                "cv_strategy": "champion_from_cv",
            }
        )
        mlflow.log_params(
            {
                "fn_cost": 10,
                "fp_cost": 1,
                "threshold_search": "valid_argmin_business_cost",
                "n_estimators": 300,
                "class_weight": "balanced",
            }
        )
        mlflow.log_metrics(
            {
                "valid_auc": float(mstar["auc"]),
                "threshold_opt": float(thr),
                "valid_business_cost_at_0.5": float(m05["business_cost"]),
                "valid_business_cost_at_opt": float(mstar["business_cost"]),
                "valid_fn_at_opt": float(mstar["fn"]),
                "valid_fp_at_opt": float(mstar["fp"]),
                "valid_accuracy_at_opt": float(mstar["accuracy"]),
                "valid_recall_at_opt": float(mstar["recall"]),
                "valid_f1_at_opt": float(mstar["f1"]),
            }
        )
        mlflow.log_artifact(str(out), artifact_path="threshold")
        mlflow.lightgbm.log_model(lgb, name="model")
    print("mlflow run logged", flush=True)


if __name__ == "__main__":
    main()
