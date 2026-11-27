"""
Explicabilité SHAP sur le champion LightGBM Optuna.

Exigence mission : importance GLOBALE (toutes demandes) + LOCALE (1 client).
Artefacts → reports/shap/ + run MLflow stage=shap.

Usage : uv run python -m src.run_shap_lgbm
Prérequis : reports/optuna_lgbm_best.json (via run_optuna_lgbm)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from src.modeling import prepare_tree_frame

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "shap"
MODELS = ROOT / "models"
BEST_JSON = ROOT / "reports" / "optuna_lgbm_best.json"
RANDOM_STATE = 42
# SHAP sur tout valid serait lent → sous-échantillon pour le summary global.
SHAP_SAMPLE_N = 1500


def _load_best_params() -> tuple[dict, float]:
    """Relit hyperparams + seuil* issus d'Optuna."""
    if not BEST_JSON.exists():
        raise SystemExit(
            f"Manquant: {BEST_JSON}. Lancer d'abord: uv run python -m src.run_optuna_lgbm"
        )
    payload = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    return payload["params"], float(payload["threshold_opt_valid"])


def _pick_local_indices(proba: np.ndarray, threshold: float) -> dict[str, int]:
    """3 dossiers types pour le chargé d'études (waterfall)."""
    refuse_idx = int(np.argmax(proba))  # risque max → refus clair
    accept_idx = int(np.argmin(proba))  # risque min → accord clair
    near_idx = int(np.argmin(np.abs(proba - threshold)))  # borderline
    return {
        "refuse_high_risk": refuse_idx,
        "accept_low_risk": accept_idx,
        "near_threshold": near_idx,
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    params, threshold = _load_best_params()
    meta = json.loads((PROC / "meta.json").read_text(encoding="utf-8"))
    train = pd.read_parquet(PROC / "train.parquet")
    valid = pd.read_parquet(PROC / "valid.parquet")

    y_train = train["TARGET"].astype(int)
    y_valid = valid["TARGET"].astype(int)
    ids_valid = valid["SK_ID_CURR"].to_numpy()
    X_train = train.drop(columns=["TARGET", "SK_ID_CURR"])
    X_valid = valid.drop(columns=["TARGET", "SK_ID_CURR"])
    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    X_tr = prepare_tree_frame(X_train, cat_cols)
    X_va = prepare_tree_frame(X_valid, cat_cols)
    for c in cat_cols:
        X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

    # --- 1) Re-fit du champion (mêmes params Optuna) ---
    print("Fit LightGBM champion (params Optuna)...", flush=True)
    model = LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_train, categorical_feature=cat_cols)
    proba_valid = model.predict_proba(X_va)[:, 1]

    model_path = MODELS / "champion_lgbm_optuna.txt"
    model.booster_.save_model(str(model_path))
    print(f"Modele sauvegarde: {model_path}", flush=True)

    # --- 2) SHAP GLOBAL : quelles features comptent en moyenne ? ---
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(SHAP_SAMPLE_N, len(X_va))
    sample_idx = rng.choice(len(X_va), size=n, replace=False)
    X_sample = X_va.iloc[sample_idx]

    print(f"SHAP TreeExplainer sur n={n} lignes valid...", flush=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Binaire LightGBM : parfois [shap_classe0, shap_classe1]
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]  # on explique la proba de DÉFAUT
    else:
        shap_pos = shap_values

    # Importance = moyenne des |contributions SHAP|
    mean_abs = np.abs(shap_pos).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp_path = REPORTS / "shap_global_importance.csv"
    importance.to_csv(imp_path, index=False)
    print("Top 10 features (mean |SHAP|):", flush=True)
    print(importance.head(10).to_string(index=False), flush=True)

    plt.figure()
    shap.summary_plot(shap_pos, X_sample, show=False, max_display=20)
    summary_path = REPORTS / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_pos, X_sample, plot_type="bar", show=False, max_display=20)
    bar_path = REPORTS / "shap_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close()

    # --- 3) SHAP LOCAL : pourquoi CE dossier a ce score ? ---
    local_ids = _pick_local_indices(proba_valid, threshold)
    local_rows = []
    for label, idx in local_ids.items():
        row = X_va.iloc[[idx]]
        sv = explainer.shap_values(row)
        if isinstance(sv, list):
            sv_pos = sv[1][0]
        else:
            sv_pos = sv[0]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base_val = float(base[1] if len(np.atleast_1d(base)) > 1 else base)
        else:
            base_val = float(base)

        # Waterfall = contribution feature par feature au score du client
        explanation = shap.Explanation(
            values=sv_pos,
            base_values=base_val,
            data=row.iloc[0].to_numpy(),
            feature_names=list(row.columns),
        )
        plt.figure()
        shap.plots.waterfall(explanation, max_display=15, show=False)
        wf_path = REPORTS / f"shap_waterfall_{label}.png"
        plt.tight_layout()
        plt.savefig(wf_path, dpi=140, bbox_inches="tight")
        plt.close()

        local_df = (
            pd.DataFrame(
                {
                    "feature": row.columns,
                    "shap_value": sv_pos,
                    "feature_value": row.iloc[0].to_numpy(),
                }
            )
            .assign(abs_shap=lambda d: d["shap_value"].abs())
            .sort_values("abs_shap", ascending=False)
        )
        local_csv = REPORTS / f"shap_local_{label}.csv"
        local_df.head(20).to_csv(local_csv, index=False)

        decision = "REFUS" if proba_valid[idx] >= threshold else "ACCORD"
        local_rows.append(
            {
                "case": label,
                "sk_id_curr": int(ids_valid[idx]),
                "y_true": int(y_valid.iloc[idx]),
                "proba_default": float(proba_valid[idx]),
                "threshold": threshold,
                "decision": decision,
                "top_feature": local_df.iloc[0]["feature"],
                "top_shap": float(local_df.iloc[0]["shap_value"]),
                "waterfall": str(wf_path.relative_to(ROOT)),
            }
        )
        print(
            f"  local {label}: SK_ID={ids_valid[idx]} proba={proba_valid[idx]:.3f} "
            f"-> {decision} | top={local_df.iloc[0]['feature']}",
            flush=True,
        )

    cases_path = REPORTS / "shap_local_cases.json"
    cases_path.write_text(json.dumps(local_rows, indent=2), encoding="utf-8")

    # --- 4) Log MLflow (figures = artefacts) ---
    tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-scoring-fe_v1")

    with mlflow.start_run(run_name="lightgbm_shap_fe_v1") as run:
        mlflow.set_tags(
            {
                "fe_version": meta["fe_version"],
                "model_family": "lightgbm",
                "stage": "shap",
            }
        )
        mlflow.log_params(
            {
                **{f"best_{k}": v for k, v in params.items()},
                "threshold_opt_valid": threshold,
                "shap_sample_n": n,
                "fn_cost": 10,
                "fp_cost": 1,
            }
        )
        mlflow.log_metrics(
            {
                "valid_auc_ref": float(
                    json.loads(BEST_JSON.read_text(encoding="utf-8"))["valid_auc"]
                ),
                "threshold_opt_valid": threshold,
            }
        )
        for p in [
            imp_path,
            summary_path,
            bar_path,
            cases_path,
            *REPORTS.glob("shap_waterfall_*.png"),
            *REPORTS.glob("shap_local_*.csv"),
        ]:
            mlflow.log_artifact(str(p), artifact_path="shap")
        try:
            mlflow.lightgbm.log_model(model, name="model")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] log_model: {exc}", flush=True)

        print(f"MLflow run: {run.info.run_id}", flush=True)
        (REPORTS / "shap_mlflow_run_id.txt").write_text(run.info.run_id, encoding="utf-8")

    print(f"Artefacts SHAP: {REPORTS}", flush=True)


if __name__ == "__main__":
    main()
