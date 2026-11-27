# Handoff — Projet 7 « Prêt à dépenser »

Date : **2026-11-25**  
À lire en premier pour un nouvel agent / une nouvelle discussion.

## Objectif mission (rappel)
**Partie 1 :** scoring crédit + coût FN/FP + MLflow + SHAP + registry.  
**Partie 2 :** Git public, API + Docker + CI/CD, logs/drift, optimisation post-déploiement.

## Où on en est

| Bloc | Statut |
|------|--------|
| Partie 1 (FE → Optuna → SHAP → Registry) | **Fait** |
| Git lean + push branche `mlops2` (ex-`EDA`) | **Fait** (D-019) |
| API FastAPI + validation + tests | **Fait** (D-018) |
| Dockerfile (slim) + run local | **Fait** (port 8001) |
| GitHub Actions CI/CD | **Fait** (D-020) |
| Logs + drift Evidently | **Fait** (D-021) |
| Optimisation ONNX / profiling | **Fait** (D-022) |

## Champion
- LightGBM Optuna — seuil\* **0.52**, coût\* ≈ **29962**
- Runtime API défaut : **ONNX** (`models/champion_lgbm_optuna.onnx`)
- Fallback : `SCORING_RUNTIME=lgbm` + `models/champion_lgbm_optuna.txt`

## Commandes utiles
```powershell
cd "c:\Users\Mathieu\Desktop\Formation\Projet 7"
uv run python -m src.run_optimize_inference
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
uv run pytest tests/test_api.py -q
# Rapport opti : reports/optimization/optimization_report.md
```

## Prochaine étape
1. (Option) README d’exécution bout-en-bout
2. (Option) merger `mlops2` → `main` ; confirmer repo public

## Contraintes agent
- Consigne + Roadmap/Decision à jour ; **NON CONFORME** si exigence manquante.
- Pas de commit de `data/` / `mlruns/` / `logs/`.
