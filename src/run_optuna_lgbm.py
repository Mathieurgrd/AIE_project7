"""
Optuna sur LightGBM (champion CV) + re-optimisation du seuil métier.

Protocole (D-015) :
  1) Optuna maximise l'AUC CV (3 folds) — ranking stable, indépendant du seuil
  2) Meilleurs params → OOF 5-fold + refit train
  3) Seuil* = argmin 10*FN+FP sur holdout valid
  4) Run MLflow parent + trials nestés + artefacts JSON/CSV

Usage :
  uv run python -m src.run_optuna_lgbm
  uv run python -m src.run_optuna_lgbm --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.modeling import (
    FN_COST_DEFAULT,
    FP_COST_DEFAULT,
    eval_at_threshold,
    optimize_threshold,
    prepare_tree_frame,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RANDOM_STATE = 42
FN_COST = FN_COST_DEFAULT
FP_COST = FP_COST_DEFAULT


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna LightGBM + seuil metier")
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--n-splits-search", type=int, default=3, help="Folds pendant Optuna (rapide)")
    p.add_argument("--n-splits-final", type=int, default=5, help="Folds evaluation finale")
    return p.parse_args()


def _oof_auc_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    params: dict,
    n_splits: int,
) -> tuple[float, float, np.ndarray]:
    """CV LightGBM : retourne (auc_mean, auc_std, proba OOF)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []

    for tr_idx, va_idx in cv.split(X, y):
        X_tr = prepare_tree_frame(X.iloc[tr_idx], cat_cols)
        X_va = prepare_tree_frame(X.iloc[va_idx], cat_cols)
        for c in cat_cols:
            X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

        model = LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y.iloc[tr_idx], categorical_feature=cat_cols)
        proba = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba
        fold_aucs.append(float(roc_auc_score(y.iloc[va_idx], proba)))

    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), oof


def _suggest_params(trial: optuna.Trial) -> dict:
    """Espace de recherche borné (solution simple, pas une grille infinie)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }


def _log_threshold_curve(curve: pd.DataFrame, title: str) -> None:
    """Courbe coût vs seuil → artefact MLflow."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "threshold_cost_curve.csv"
        curve.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path), artifact_path="threshold")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(curve["threshold"], curve["business_cost"], color="#1f4e79")
        best = curve.loc[curve["business_cost"].idxmin()]
        ax.axvline(
            best["threshold"],
            color="#c44e52",
            linestyle="--",
            label=f"seuil*={best['threshold']:.2f}",
        )
        ax.set_xlabel("Seuil")
        ax.set_ylabel(f"Cout metier ({FN_COST}*FN + {FP_COST}*FP)")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        png_path = tmp_path / "threshold_cost_curve.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(png_path), artifact_path="threshold")


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

    tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-scoring-fe_v1")

    print(
        f"Optuna LGBM | trials={args.n_trials} | "
        f"search_folds={args.n_splits_search} | final_folds={args.n_splits_final}",
        flush=True,
    )

    trial_rows: list[dict] = []

    with mlflow.start_run(run_name="lightgbm_optuna_fe_v1") as parent:
        mlflow.set_tags(
            {
                "fe_version": meta["fe_version"],
                "model_family": "lightgbm",
                "stage": "optuna",
                "cv_strategy": "StratifiedKFold",
            }
        )
        mlflow.log_params(
            {
                "n_trials": args.n_trials,
                "n_splits_search": args.n_splits_search,
                "n_splits_final": args.n_splits_final,
                "objective": "maximize_cv_auc",
                "fn_cost": FN_COST,
                "fp_cost": FP_COST,
                "class_weight": "balanced",
                "sampler": "TPE",
            }
        )

        # --- Objectif Optuna : 1 trial = 1 jeu d'hyperparams + AUC CV ---
        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial)
            auc_mean, auc_std, oof = _oof_auc_lgbm(
                X_train,
                y_train,
                cat_cols,
                params,
                n_splits=args.n_splits_search,
            )
            # Coût @0.5 loggé pour info (pas l'objectif — le seuil vient après).
            m05 = eval_at_threshold(
                y_train, oof, threshold=0.5, fn_cost=FN_COST, fp_cost=FP_COST
            )

            with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        "cv_auc_mean": auc_mean,
                        "cv_auc_std": auc_std,
                        "oof_business_cost_at_0.5": float(m05["business_cost"]),
                    }
                )

            trial.set_user_attr("cv_auc_std", auc_std)
            trial.set_user_attr("oof_cost_0.5", float(m05["business_cost"]))
            trial_rows.append(
                {
                    "trial": trial.number,
                    "cv_auc_mean": auc_mean,
                    "cv_auc_std": auc_std,
                    "oof_cost_0.5": float(m05["business_cost"]),
                    **params,
                }
            )
            print(
                f"  trial {trial.number:03d} AUC={auc_mean:.4f} (+/-{auc_std:.4f}) "
                f"cost@0.5={m05['business_cost']:.0f}",
                flush=True,
            )
            return auc_mean  # Optuna maximise cette valeur

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE),  # exploration guidée
            study_name="lgbm_fe_v1",
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best_params = dict(study.best_params)
        print(f"\nBest trial #{study.best_trial.number} AUC={study.best_value:.4f}", flush=True)
        print(f"Best params: {best_params}", flush=True)

        # --- Éval finale avec les meilleurs params ---
        print("Evaluation finale 5-fold + seuil valid...", flush=True)
        auc_mean, auc_std, oof = _oof_auc_lgbm(
            X_train, y_train, cat_cols, best_params, n_splits=args.n_splits_final
        )
        m_oof_05 = eval_at_threshold(y_train, oof, 0.5, FN_COST, FP_COST)
        thr_oof, _ = optimize_threshold(y_train, oof, FN_COST, FP_COST)
        m_oof_star = eval_at_threshold(y_train, oof, thr_oof, FN_COST, FP_COST)

        X_tr_fit = prepare_tree_frame(X_train, cat_cols)
        X_va_fit = prepare_tree_frame(X_valid, cat_cols)
        for c in cat_cols:
            X_va_fit[c] = pd.Categorical(X_va_fit[c], categories=X_tr_fit[c].cat.categories)

        final_model = LGBMClassifier(
            **best_params,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        final_model.fit(X_tr_fit, y_train, categorical_feature=cat_cols)
        proba_valid = final_model.predict_proba(X_va_fit)[:, 1]

        # Seuil métier calé sur valid (décision de prod pour ce champion)
        thr_valid, curve_valid = optimize_threshold(y_valid, proba_valid, FN_COST, FP_COST)
        m_val_05 = eval_at_threshold(y_valid, proba_valid, 0.5, FN_COST, FP_COST)
        m_val_star = eval_at_threshold(y_valid, proba_valid, thr_valid, FN_COST, FP_COST)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(
            {
                "best_cv_auc_search": float(study.best_value),
                "final_cv_auc_mean": auc_mean,
                "final_cv_auc_std": auc_std,
                "oof_business_cost_at_0.5": float(m_oof_05["business_cost"]),
                "oof_business_cost_at_opt": float(m_oof_star["business_cost"]),
                "threshold_opt_oof": float(thr_oof),
                "valid_auc": float(m_val_star["auc"]),
                "valid_business_cost_at_0.5": float(m_val_05["business_cost"]),
                "valid_business_cost_at_opt": float(m_val_star["business_cost"]),
                "threshold_opt_valid": float(thr_valid),
                "valid_fn_at_opt": float(m_val_star["fn"]),
                "valid_fp_at_opt": float(m_val_star["fp"]),
                "valid_accuracy_at_opt": float(m_val_star["accuracy"]),
                "valid_recall_at_opt": float(m_val_star["recall"]),
                "valid_f1_at_opt": float(m_val_star["f1"]),
                "baseline_valid_auc_ref": 0.7844,
                "baseline_valid_cost_ref": 30013.0,
            }
        )
        _log_threshold_curve(curve_valid, "LightGBM Optuna — cout vs seuil (valid)")

        # Artefacts locaux (lus par SHAP / Registry ensuite)
        reports = ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        trials_df = pd.DataFrame(trial_rows).sort_values("cv_auc_mean", ascending=False)
        trials_path = reports / "optuna_lgbm_trials.csv"
        trials_df.to_csv(trials_path, index=False)
        mlflow.log_artifact(str(trials_path), artifact_path="optuna")

        best_payload = {
            "fe_version": meta["fe_version"],
            "best_trial": study.best_trial.number,
            "best_cv_auc_search": float(study.best_value),
            "final_cv_auc_mean": auc_mean,
            "final_cv_auc_std": auc_std,
            "params": best_params,
            "threshold_opt_valid": float(thr_valid),
            "valid_auc": float(m_val_star["auc"]),
            "valid_business_cost_at_0.5": float(m_val_05["business_cost"]),
            "valid_business_cost_at_opt": float(m_val_star["business_cost"]),
            "fn_cost": FN_COST,
            "fp_cost": FP_COST,
        }
        best_path = reports / "optuna_lgbm_best.json"
        best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(best_path), artifact_path="optuna")

        try:
            mlflow.lightgbm.log_model(final_model, name="model")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] log_model ignore: {exc}", flush=True)
            mlflow.set_tag("model_logged", "false")

        print(
            f"VALID AUC={m_val_star['auc']:.4f} | "
            f"cost 0.5={m_val_05['business_cost']:.0f} | "
            f"seuil*={thr_valid:.2f} cost_opt={m_val_star['business_cost']:.0f}",
            flush=True,
        )
        print(f"Artifacts: {trials_path.name}, {best_path.name}", flush=True)
        print(f"MLflow parent run: {parent.info.run_id}", flush=True)


if __name__ == "__main__":
    main()
