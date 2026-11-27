"""
Enregistre le champion dans le MLflow Model Registry + smoke predict.

Pourquoi un wrapper pyfunc ?
  LightGBM + dtypes category cassent le schema MLflow natif au serving.
  CreditLGBMWrapper recharge le booster et reconvertit les cats à l'inférence.

Usage : uv run python -m src.run_register_champion
Serving ensuite :
  uv run mlflow models serve -m "models:/credit-scoring-lgbm@champion" \\
      --port 5001 --env-manager local --host 127.0.0.1
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

from src.modeling import prepare_tree_frame

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
BEST_JSON = ROOT / "reports" / "optuna_lgbm_best.json"
MODEL_NAME = "credit-scoring-lgbm"
RANDOM_STATE = 42


class CreditLGBMWrapper(PythonModel):
    """
    Modèle MLflow « prêt à servir ».
    load_context : charge le fichier booster LightGBM
    predict      : renvoie les proba de défaut (le seuil 0.52 reste côté appli)
    """

    def __init__(self, cat_cols: list[str]):
        self.cat_cols = cat_cols

    def load_context(self, context):
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=context.artifacts["booster"])

    def _prepare(self, model_input: pd.DataFrame) -> pd.DataFrame:
        X = model_input.copy()
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X

    def predict(self, context, model_input, params=None):
        X = self._prepare(pd.DataFrame(model_input))
        return self.model.predict(X)  # proba classe positive (défaut)


def main() -> None:
    # --- 1) Recharger params Optuna + données ---
    payload = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    params = payload["params"]
    threshold = float(payload["threshold_opt_valid"])

    train = pd.read_parquet(PROC / "train.parquet")
    valid = pd.read_parquet(PROC / "valid.parquet")
    y_train = train["TARGET"].astype(int)
    X_train = train.drop(columns=["TARGET", "SK_ID_CURR"])
    X_valid = valid.drop(columns=["TARGET", "SK_ID_CURR"])
    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    X_tr = prepare_tree_frame(X_train, cat_cols)
    X_va = prepare_tree_frame(X_valid, cat_cols)
    for c in cat_cols:
        X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

    # --- 2) Fit + sauver le booster (artefact servi) ---
    model = LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_train, categorical_feature=cat_cols)

    booster_path = ROOT / "models" / "champion_lgbm_optuna.txt"
    booster_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(booster_path))

    # Exemple d'entrée en strings (friendly pour le schema HTTP)
    sample = X_va.head(3).copy()
    for c in cat_cols:
        sample[c] = sample[c].astype(str)
    proba_sample = model.predict_proba(X_va.head(3))[:, 1]
    signature = infer_signature(sample, proba_sample)

    # --- 3) Log + register dans MLflow ---
    tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-scoring-fe_v1")
    client = MlflowClient()

    with mlflow.start_run(run_name="lightgbm_register_fe_v1") as run:
        mlflow.set_tags(
            {
                "fe_version": "fe_v1",
                "model_family": "lightgbm",
                "stage": "registry",
            }
        )
        mlflow.log_params({**{f"best_{k}": v for k, v in params.items()}, "threshold_opt_valid": threshold})
        mlflow.log_metrics(
            {
                "valid_auc": float(payload["valid_auc"]),
                "valid_business_cost_at_opt": float(payload["valid_business_cost_at_opt"]),
                "threshold_opt_valid": threshold,
            }
        )
        mlflow.pyfunc.log_model(
            name="model",
            python_model=CreditLGBMWrapper(cat_cols=cat_cols),
            artifacts={"booster": str(booster_path)},
            signature=signature,
            input_example=sample,
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id
        print(f"Logged + registered name={MODEL_NAME} run_id={run_id}", flush=True)

    # --- 4) Staging + alias @champion (URI stable pour le serving) ---
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Staging",
            archive_existing_versions=True,
        )
        stage_or_alias = "Staging"
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] stage: {exc}", flush=True)
        stage_or_alias = f"v{latest.version}"
    try:
        client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] alias: {exc}", flush=True)

    print(f"Model {MODEL_NAME} v{latest.version} -> {stage_or_alias} (+ alias champion)", flush=True)

    # --- 5) Smoke test local (sans serveur HTTP) ---
    model_uri = f"models:/{MODEL_NAME}@champion"
    loaded = mlflow.pyfunc.load_model(model_uri)
    smoke = X_va.head(5).copy()
    for c in cat_cols:
        smoke[c] = smoke[c].astype(str)
    pred = np.asarray(loaded.predict(smoke), dtype=float)
    print("Smoke predict proba (5 lignes):", np.round(pred, 4), flush=True)
    print(f"Seuil metier cote appli: {threshold}", flush=True)

    out = ROOT / "reports" / "registry_champion.json"
    out.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "version": latest.version,
                "stage": stage_or_alias,
                "alias": "champion",
                "run_id": run_id,
                "threshold_opt_valid": threshold,
                "model_uri_alias": model_uri,
                "model_uri_stage": f"models:/{MODEL_NAME}/Staging",
                "smoke_proba_head5": [float(x) for x in pred],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Ecrit: {out}", flush=True)


if __name__ == "__main__":
    main()
