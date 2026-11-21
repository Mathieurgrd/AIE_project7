# Handoff — Projet 7 « Prêt à dépenser »

Date : **2026-10-09**  
À lire en premier pour un nouvel agent / une nouvelle discussion.

## Objectif mission (rappel)
Scoring crédit (proba de défaut) + coût métier FN/FP + MLOps MLflow (tracking → registry → serving) + explicabilité globale/locale.

## Où on en est
**Fin de mission technique principale.** FE → CV → Optuna → seuil → SHAP → Registry + serving testé.  
Reste surtout polish livrable (README, notebooks) et récit équité éventuel.

| Bloc | Statut |
|------|--------|
| Env UV / Python 3.11 | Fait |
| EDA + FE **fe_v1** | Fait + validé Mathieu |
| MLflow tracking + UI | Fait (`sqlite:///mlflow.db`) |
| StratifiedKFold + multi-algos | Fait (D-013) |
| Optuna hyperparams | **Fait** (D-015) — ne pas croire « enlevé » |
| Seuil métier | Fait (0.52) |
| SHAP global + local | **Fait** (D-016) |
| Model Registry + serving | **Fait** (D-017, testé HTTP) |

## Champion
- LightGBM Optuna — valid AUC ≈ **0.786**, seuil\* **0.52**, coût\* ≈ **29962**
- Registry : `models:/credit-scoring-lgbm@champion`
- Récit complet : **`docs/demarches_projet.txt`**

## Commandes
Voir `docs/commandes_utiles.txt`

```powershell
cd "c:\Users\Mathieu\Desktop\Formation\Projet 7"
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
uv run python -m src.run_shap_lgbm
uv run python -m src.run_register_champion
uv run mlflow models serve -m "models:/credit-scoring-lgbm@champion" --port 5001 --env-manager local --host 127.0.0.1
```

## Prochaine étape recommandée
1. Relire figures `reports/shap/` + `docs/demarches_projet.txt` (discours métier)
2. README d’exécution bout-en-bout
3. (Option) discuter exclusion `CODE_GENDER` / fe_v2

## Contraintes agent
- Consigne + Roadmap/Decision à jour ; **NON CONFORME** si exigence manquante.
- Pas de git sauf demande explicite.
