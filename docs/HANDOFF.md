# Handoff — Projet 7 « Prêt à dépenser »

Date : **2026-11-21**  
À lire en premier pour un nouvel agent / une nouvelle discussion.

## Objectif mission (rappel)
**Partie 1 :** scoring crédit + coût FN/FP + MLflow + SHAP + registry.  
**Partie 2 :** Git public, API + Docker + CI/CD, logs/drift, optimisation post-déploiement.

## Où on en est

| Bloc | Statut |
|------|--------|
| Partie 1 (FE → Optuna → SHAP → Registry) | **Fait** |
| Git lean + push branche `EDA` | **Fait** (D-019) — `main` distant encore = commit initial |
| API FastAPI + validation + tests | **Fait** (D-018) — 7 tests OK |
| Dockerfile | **Fait** (validé local port 8001) |
| GitHub Actions CI/CD | **Fait** (D-020) — vérifier run vert sur le repo |
| Drift / logs analyse | Base log JSONL OK ; analyse Evidently **à faire** |
| Optimisation ONNX / profiling | **À faire** |

## Champion
- LightGBM Optuna — seuil\* **0.52**, coût\* ≈ **29962**
- Fichier servi par l’API : `models/champion_lgbm_optuna.txt`
- Schéma : `reports/api_feature_schema.json`

## Commandes API
```powershell
cd "c:\Users\Mathieu\Desktop\Formation\Projet 7"
uv sync
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
# Swagger : http://127.0.0.1:8000/docs
uv run pytest tests/test_api.py -q
```

## Prochaine étape
1. Vérifier l’onglet **Actions** sur GitHub (run `CI API` vert)
2. Script/notebook **drift** (Evidently) — étape 3 consigne
3. (Option) merger `EDA` → `main` ; profiling / ONNX — étape 4

## Contraintes agent
- Consigne + Roadmap/Decision à jour ; **NON CONFORME** si exigence manquante.
- Pas de git push destructeur ; pas de commit de `data/` / `mlruns/`.
