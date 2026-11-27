"""
Package source — Projet 7 « Prêt à dépenser ».

Modules :
  features              → feature engineering fe_v1 (aggs multi-tables)
  build_dataset         → génère data/interim + data/processed
  modeling              → coût métier, seuil, estimateurs CV
  run_cv_baselines      → StratifiedKFold multi-modèles → MLflow
  run_optuna_lgbm       → Optuna LightGBM + seuil valid
  run_threshold_on_valid→ seuil* seul sur LGBM baseline
  run_shap_lgbm         → explicabilité SHAP global / local
  run_register_champion → Model Registry + smoke serving
"""

__all__ = [
    "features",
    "build_dataset",
    "modeling",
    "run_cv_baselines",
    "run_optuna_lgbm",
    "run_threshold_on_valid",
    "run_shap_lgbm",
    "run_register_champion",
]
