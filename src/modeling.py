"""
Boîte à outils commune au scoring crédit.

Rôle : métriques métier (coût FN/FP), optimisation du seuil,
       preprocessing sklearn, constructeurs de modèles pour la CV.
Utilisé par : run_cv_baselines, run_optuna_lgbm, run_shap_*, etc.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# --- Coût métier (hypothèse consigne / D-011) ---
# FN = mauvais client accepté (très cher) ; FP = bon client refusé (moins cher).
FN_COST_DEFAULT = 10
FP_COST_DEFAULT = 1


def business_cost(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    fn_cost: float = FN_COST_DEFAULT,
    fp_cost: float = FP_COST_DEFAULT,
) -> tuple[float, int, int]:
    """Coût total = fn_cost * FN + fp_cost * FP. Retourne (coût, n_fn, n_fp)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())  # défaut raté
    fp = int(((y_true == 0) & (y_pred == 1)).sum())  # bon client refusé
    return float(fn_cost * fn + fp_cost * fp), fn, fp


def eval_at_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    threshold: float = 0.5,
    fn_cost: float = FN_COST_DEFAULT,
    fp_cost: float = FP_COST_DEFAULT,
) -> dict[str, float | int]:
    """
    Passe des probas → classes via un seuil, puis calcule métriques tech + métier.
    C'est ce qu'on logue ensuite dans MLflow (auc, business_cost, fn, fp, …).
    """
    pred = (np.asarray(proba) >= threshold).astype(int)
    cost, fn, fp = business_cost(y_true, pred, fn_cost=fn_cost, fp_cost=fp_cost)
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, proba)),  # indépendant du seuil
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "business_cost": float(cost),
        "fn": fn,
        "fp": fp,
    }


def optimize_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    fn_cost: float = FN_COST_DEFAULT,
    fp_cost: float = FP_COST_DEFAULT,
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Balaye des seuils de décision et garde celui qui minimise le coût métier.
    → c'est le « seuil* » de la consigne (ne pas rester à 0.5 par défaut).

    Returns
    -------
    best_threshold, curve_df
    """
    if thresholds is None:
        # Pas 0.01 trop fin inutile ; 0.05–0.95 couvre la consigne 0.1–0.9.
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rows: list[dict[str, float | int]] = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        cost, fn, fp = business_cost(y_true, pred, fn_cost=fn_cost, fp_cost=fp_cost)
        rows.append(
            {
                "threshold": float(t),
                "business_cost": float(cost),
                "fn": fn,
                "fp": fp,
                "accuracy": float(accuracy_score(y_true, pred)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
            }
        )
    curve = pd.DataFrame(rows)
    best_idx = int(curve["business_cost"].idxmin())
    best_threshold = float(curve.loc[best_idx, "threshold"])
    return best_threshold, curve


def make_sklearn_preprocess(num_cols: list[str], cat_cols: list[str]):
    """
    Pipeline anti-leakage pour LR / RF :
    num → médiane + scale ; cat → mode + One-Hot.
    À fitter UNIQUEMENT sur le fold train (jamais sur valid).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def prepare_tree_frame(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Pour LightGBM / XGBoost : colonnes texte → dtype category (natif)."""
    out = X.copy()
    for c in cat_cols:
        out[c] = out[c].astype("category")
    return out


def build_estimators(
    num_cols: list[str],
    cat_cols: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Catalogue des 4 familles (consigne étape 3).

    Chaque entrée contient :
      - estimator : objet sklearn / LGBM / XGB (non encore fit)
      - needs_sklearn_preprocess : True si impute+OHE nécessaires
      - params : hyperparams à logger dans MLflow
      - fit_kwargs : args supplémentaires au .fit() (ex. categorical_feature)
    """
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    preprocess = make_sklearn_preprocess(num_cols, cat_cols)

    return {
        # Baseline linéaire : rapide, interprétable (coeffs), besoin de preprocess.
        "logreg": {
            "family": "logreg",
            "needs_sklearn_preprocess": True,
            "estimator": Pipeline(
                steps=[
                    ("preprocess", preprocess),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",  # gère le ~8 % de défauts
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "params": {"class_weight": "balanced", "max_iter": 1000},
        },
        # Forêt : famille « bagging » demandée par la consigne.
        "random_forest": {
            "family": "random_forest",
            "needs_sklearn_preprocess": True,
            "estimator": Pipeline(
                steps=[
                    ("preprocess", make_sklearn_preprocess(num_cols, cat_cols)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=12,
                            min_samples_leaf=20,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
            },
        },
        # Champion provisoire tabulaire : cats + NA natives, class_weight.
        "lightgbm": {
            "family": "lightgbm",
            "needs_sklearn_preprocess": False,
            "estimator": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "class_weight": "balanced",
            },
            "fit_kwargs": {"categorical_feature": cat_cols},
        },
        # Boosting concurrent ; déséquilibre via scale_pos_weight (posé au fit).
        "xgboost": {
            "family": "xgboost",
            "needs_sklearn_preprocess": False,
            "estimator": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                enable_categorical=True,
                random_state=random_state,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "enable_categorical": True,
                "imbalance": "scale_pos_weight",
            },
            "fit_kwargs": {},
        },
    }
