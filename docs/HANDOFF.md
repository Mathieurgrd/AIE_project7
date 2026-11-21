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
| Dockerfile | **Fait** (non validé Docker Desktop chez Mathieu) |
| GitHub Actions CI/CD | **À faire** |
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

## Prochaine étape (besoin Mathieu ou suite agent)
1. Valider **Docker Desktop** + `docker build -t credit-scoring-api .`
2. Choisir default branch GitHub (`main` vs `EDA`) / PR
3. Ajouter **GitHub Actions** (test → build image)
4. Notebook/script **drift** (Evidently) sur `logs/predictions.jsonl`

## Contraintes agent
- Consigne + Roadmap/Decision à jour ; **NON CONFORME** si exigence manquante.
- Pas de git push destructeur ; pas de commit de `data/` / `mlruns/`.
