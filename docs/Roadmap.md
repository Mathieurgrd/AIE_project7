# Roadmap — Projet 7 « Prêt à dépenser »

Document vivant : à mettre à jour au fur et à mesure.  
Dernière màj : **2026-10-09** (SHAP + Registry + serving)

## Où en est-on ? (résumé)

**Mission technique principale couverte** : FE + MLflow + CV + Optuna + seuil + **SHAP** + **Registry/serving**.  
Reste : polish livrable (README, notebooks) ; options fe_v2 / MLP / équité CODE_GENDER.

Repère consigne OpenClassrooms :

| Bloc consigne | Statut |
|---------------|--------|
| 1 — Préparation des données | **Fait** (fe_v1) |
| 2 — Tracking MLflow + UI | **Fait** |
| 3 — Multi-algos + CV | **Fait** (LR, RF, LightGBM, XGBoost + StratifiedKFold 5) |
| 4 — Optuna/GridSearch + seuil métier | **Fait** (Optuna 25 trials + seuil\* valid = 0.52) |
| Mission : importance globale/locale | **Fait** (SHAP, D-016) |
| Mission : Model Registry + serving | **Fait** (D-017, HTTP testé) |

**Prochaine étape recommandée :** README + relecture figures SHAP ; récit dans `docs/demarches_projet.txt`.

---

## Checklist mission (contraintes Michaël / livrables)

### Objectifs globaux

| # | Exigence | Statut | Preuve / commentaire |
|---|----------|--------|----------------------|
| G1 | Modèle de scoring (proba de défaut) | **Fait** (v1) | Champion LightGBM Optuna + seuil 0.52 |
| G2 | Feature importance globale + locale | **Fait** | SHAP `reports/shap/` + run `lightgbm_shap_fe_v1` |
| G3 | MLOps bout-en-bout tracking → pré-prod | **Fait** (local) | Tracking + UI + Registry `credit-scoring-lgbm@champion` + serve testé |

### Contraintes métier / méthodo

| # | Exigence | Statut | Preuve / commentaire |
|---|----------|--------|----------------------|
| M1 | Gérer le déséquilibre des classes | Fait (v1) | `class_weight="balanced"` ; XGB `scale_pos_weight` ; SMOTE non utilisé |
| M2 | Score métier coût FN/FP | Fait | `10*FN + 1*FP` loggé (OOF + valid) |
| M3 | Optimiser le seuil (pas rester à 0.5) | **Fait** | Champion Optuna : **seuil\*=0.52** sur valid (coût 29962 < 30073@0.5) |
| M4 | Suivre AUC (+ accuracy) en parallèle | Fait | CV + valid logués MLflow |
| M5 | Cross-validation (StratifiedKFold) | **Fait** | 5 folds, seed 42 ; runs `*_cv5_seuil_opt_fe_v1` |
| M6 | Opti hyperparams (GridSearchCV / Optuna) | **Fait** | Optuna 25 trials, run `lightgbm_optuna_fe_v1` (D-015) |
| M7 | Alerte si AUC > 0.82 (overfit ?) | OK pour l’instant | Meilleur valid AUC ≈ 0.786 |

### MLOps minimum imposé

| # | Exigence | Statut | Preuve / commentaire |
|---|----------|--------|----------------------|
| O1 | Tracking d’expérimentations MLflow | Fait | Expérience `credit-scoring-fe_v1` |
| O2 | UI MLflow | Fait | `sqlite:///mlflow.db`, port 5000 |
| O3 | Model Registry | **Fait** | `credit-scoring-lgbm` v2 Staging + alias champion |
| O4 | Tester le serving MLflow | **Fait** | `POST /invocations` → proba OK |

### Kernels Kaggle (optionnel mais encouragé)

| # | Exigence | Statut | Preuve / commentaire |
|---|----------|--------|----------------------|
| K1 | S’inspirer / adapter kernels | Partiel | Documenté a posteriori (Koehrsen EDA, jsaguiar FE) ; pas de copie ; fe_v1 aligné « esprit » jsaguiar |

### Environnement / hygiène

| # | Point | Statut |
|---|-------|--------|
| E1 | Env isolé (uv / .venv) | Fait |
| E2 | Données jointes / enrichies | Fait (`data/processed/`) |
| E3 | Docs traçabilité | Fait (`choix_techniques.txt`, `docs/Decision.md`, `mlflow_cas_usage.txt`) |

---

## Jalons détaillés

### J0 — Organisation repo
- [x] Structure `data/`, `notebooks/`, `src/`, `models/`, `reports/`
- [x] Env UV + dépendances
- [x] Notes de compréhension (`notes.txt`)

### J1 — EDA (consigne étape 1, partie exploration)
- [x] Notebook `01_eda_application_train.ipynb`
- [x] TARGET ~8 %, NA, EXT_SOURCE, sentinelle `DAYS_EMPLOYED`
- [ ] (Option) EDA complémentaire catégorielles / corrélations — non bloquant

### J2 — Feature engineering (consigne étape 1, préparation)
- [x] `src/features.py` + `src/build_dataset.py` (**fe_v1**)
- [x] Aggs bureau, bureau_balance, previous, installments, POS, CC
- [x] Split stratifié 80/20 + artefacts parquet
- [x] Smoke test LightGBM AUC ~0.78
- [ ] fe_v2 éventuel (enrichissement type jsaguiar Active/Closed) — optionnel

### J3 — MLflow tracking + UI (consigne étape 2)
- [x] Backend SQLite (`mlflow.db`)
- [x] UI opérationnelle
- [x] Runs LR + LightGBM visibles (params + métriques + modèles)

### J4 — Modélisation multi-algos + CV (consigne étape 3) ← **fait**
- [x] Logistic Regression
- [x] LightGBM
- [x] Random Forest
- [x] XGBoost
- [ ] (Option) MLP
- [x] StratifiedKFold / évaluation robuste (`src/run_cv_baselines.py`)
- [x] Comparaison systématique sous coût métier + AUC  
  **Ordre valid coût@0.5 :** LGBM 30013 < XGB 30421 < LR 31666 < RF 32324  
  **CV AUC mean :** LGBM 0.779 > XGB 0.774 > LR 0.768 > RF 0.757

### J5 — Optimisation + seuil (consigne étape 4) ← **fait**
- [x] Optuna sur LightGBM (`src/run_optuna_lgbm.py`, 25 trials)
- [x] Courbe coût vs seuil ; seuil* optimal (OOF + valid champion Optuna → **0.52**)
- [x] Choix courant justifié : LGBM Optuna (AUC 0.786, coût* 29962) > baseline LGBM / XGB / LR / RF
- [x] Ratio FN:FP : solution **simple** 10:1 (D-011)

### J6 — Interprétabilité (mission) ← **fait**
- [x] Importance globale (SHAP summary + bar + CSV)
- [x] Importance locale (3 waterfalls : refus / accord / proche seuil)

### J7 — MLOps pré-prod (mission Michaël) ← **fait**
- [x] Enregistrer le meilleur modèle dans le **Model Registry**
- [x] Tester **serving** MLflow (requête → proba)
- [x] Documenter UI / serve / charger (`docs/commandes_utiles.txt`, `demarches_projet.txt`)

### J8 — Livrable final
- [ ] README d’exécution bout-en-bout
- [ ] Notebooks ordonnés / nettoyés
- [x] Roadmap + Decision à jour (2026-10-09)
- [x] Récit démarches `docs/demarches_projet.txt`

---

## Conformité : écarts ouverts (à traiter)

1. ~~**Seuil à 0.5**~~ — **levé** (D-014 / D-015).  
2. ~~**Pas de cross-validation**~~ — **levé** (D-013).  
3. ~~**Peu d’algorithmes**~~ — **levé**.  
4. ~~**Pas d’explicabilité**~~ — **levé** (D-016).  
5. ~~**Registry + serving**~~ — **levé** (D-017).  
6. ~~**Hyperparams**~~ — **levé** (D-015).  

Écarts mineurs / optionnels : MLP non testé ; README final ; monitoring équité `CODE_GENDER`.

---

## Historique d’avancement (bref)

| Date | Avancement |
|------|------------|
| 2026-09-30 | Prise de notes, compréhension mission + raw |
| 2026-10-02 | EDA notebook 01 |
| 2026-10-07 | fe_v1, MLflow SQLite UI, baselines LR/LGBM, docs cas MLflow |
| 2026-10-07 | Création `docs/Roadmap.md` + `docs/Decision.md` |
| 2026-10-07 | StratifiedKFold 5 + RF/XGB + seuil métier (OOF + valid) ; résumé `reports/cv_threshold_summary.csv` |
| 2026-10-07 | Optuna LightGBM 25 trials ; valid AUC≈0.786, seuil\*=0.52, coût\*≈29962 ; `reports/optuna_lgbm_best.json` |
| 2026-10-09 | SHAP global+local ; Registry `credit-scoring-lgbm@champion` ; serving HTTP OK ; `docs/demarches_projet.txt` |
