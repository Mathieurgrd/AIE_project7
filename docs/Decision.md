# Decision log — Projet 7 « Prêt à dépenser »

Journal des **choix techniques / design** et de leurs justifications.  
Format : une entrée = une décision. Les plus anciennes en bas éventuellement ; ici chronologique croissant.

Complément détaillé FE : `choix_techniques.txt` (racine).  
Complément MLflow pédagogique : `mlflow_cas_usage.txt` (racine).

Dernière màj : **2026-10-09** (SHAP + Registry)

---

## Comment lire ce fichier

| Champ | Sens |
|-------|------|
| **Statut** | Accepté / Provisoire / Remplacé |
| **Contexte** | Pourquoi on devait trancher |
| **Décision** | Ce qu’on a choisi |
| **Pourquoi** | Justification simple |
| **Conséquences** | Ce que ça implique pour la suite |

---

## D-001 — Environnement : UV + Python 3.11

- **Statut :** Accepté  
- **Contexte :** Isoler les deps (MLflow, LightGBM, etc.) sans conflits système.  
- **Décision :** Projet géré avec `uv`, Python pin `3.11`, venv local `.venv`.  
- **Pourquoi :** Reproductibilité et consigne MLflow (« environnement isolé »).  
- **Conséquences :** Lancer via `uv run …` ; `uv.lock` à conserver.

---

## D-002 — Organisation code : notebooks + `src/`

- **Statut :** Accepté  
- **Contexte :** EDA exploratoire vs FE rejouable.  
- **Décision :** EDA / narration en notebooks ; FE industrialisé dans `src/features.py` + `src/build_dataset.py`.  
- **Pourquoi :** Éviter un FE fragile « cellule par cellule » ; même pipeline train/test.  
- **Conséquences :** Régénérer les datasets avec `uv run python -m src.build_dataset`.

---

## D-003 — Feature engineering `fe_v1` (périmètre)

- **Statut :** Accepté (validé Mathieu 2026-10-07)  
- **Contexte :** Tables Home Credit 1→N à agréger sans leakage.  
- **Décision :** Clean `application_*` + aggs `bureau`, `bureau_balance`, `previous_application`, `installments_payments`, `POS_CASH`, `credit_card` ; merge left sur `SK_ID_CURR` ; ~220 colonnes.  
- **Pourquoi :** Le signal de défaut est surtout dans l’historique de crédit/paiement ; version allégée mais complète pour démarrer modèles + MLOps.  
- **Conséquences :** Artefacts dans `data/interim/` et `data/processed/` ; tag MLflow `fe_version=fe_v1`.

---

## D-004 — Anti-leakage preprocessing

- **Statut :** Accepté  
- **Contexte :** Imputation / OHE / scaling ne doivent pas « voir » le valid.  
- **Décision :** Avant split = transformations déterministes seulement (âge, ratios, groupby). Fit impute/OHE/scaler **après** split (pipeline LR). Trees (LightGBM) : catégories + NA natives.  
- **Pourquoi :** Sinon fuite d’info valid → scores trop optimistes.  
- **Conséquences :** Ne jamais scaler/imputer sur train+valid mélangés.

---

## D-005 — Traitements issus de l’EDA

- **Statut :** Accepté  
- **Décisions & pourquoi :**
  - `DAYS_EMPLOYED == 365243` → NaN + flag : code sentinelle, pas une ancienneté réelle.  
  - Ratios crédit/revenu, annuité/revenu, crédit/prix bien : charge relative > montants bruts.  
  - `HOUSING_MISSING` : NA logement souvent en bloc ; l’absence d’info peut être un signal.  
  - Garder `EXT_SOURCE_*` : meilleurs séparateurs univariés observés.  
  - Ne pas dropper massivement les colonnes >60 % NA : arbres + flags suffisent en v1.

---

## D-006 — Split stratifié 80/20, seed 42

- **Statut :** Accepté  
- **Contexte :** TARGET déséquilibré (~8 %).  
- **Décision :** `train_test_split(..., stratify=TARGET, test_size=0.2, random_state=42)`.  
- **Pourquoi :** Garder le même taux de défaut train/valid.  
- **Conséquences :** Holdout `valid` pour confirmation ; robustesse via **D-013** (StratifiedKFold sur train).

---

## D-007 — Kernels Kaggle

- **Statut :** Accepté (inspiration, pas copie)  
- **Contexte :** Consigne optionnelle (Koehrsen EDA, jsaguiar FE).  
- **Décision :** fe_v1 conçu d’abord sans lecture ligne à ligne ; rapprochement documenté ensuite (esprit jsaguiar : aggs + ratios + retards, version allégée).  
- **Pourquoi :** Auditable / pédagogique ; focus mission OC (coût métier, MLOps) plutôt que max leaderboard.  
- **Conséquences :** fe_v2 possible si besoin d’aggs Active/Closed plus fines.

---

## D-008 — Tracking MLflow : SQLite (pas file store)

- **Statut :** Accepté  
- **Contexte :** MLflow récent refuse `./mlruns` file store (maintenance).  
- **Décision :** `sqlite:///mlflow.db` pour UI et tracking ; expérience `credit-scoring-fe_v1`.  
- **Pourquoi :** Backend supporté, simple en local.  
- **Conséquences :** Commande UI :  
  `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`

---

## D-009 — Baselines modèles

- **Statut :** Accepté (étape intermédiaire)  
- **Décision :** Logistic Regression (pipeline impute+scale+OHE) + LightGBM (`class_weight=balanced`).  
- **Pourquoi :** Baseline linéaire vs boosting tabulaire ; démarre la comparaison MLflow.  
- **Résultats @ seuil 0.5 (coût 10:1) :** LGBM AUC≈0.784, coût≈30013 ; LR AUC≈0.770, coût≈31666.  
- **Conséquences :** Encore ajouter RF/XGBoost + CV avant de figer le champion.

---

## D-010 — Déséquilibre des classes

- **Statut :** Accepté (v1)  
- **Décision :** `class_weight="balanced"` (pas SMOTE pour l’instant).  
- **Pourquoi :** Simple, pas de leakage lié au resampling hors folds ; conforme « méthode au choix ».  
- **Conséquences :** SMOTE éventuel **dans** les folds CV seulement, si besoin.

---

## D-011 — Coût métier FN:FP

- **Statut :** Provisoire (hypothèse de travail)  
- **Décision :** Utiliser **10:1** (exemple consigne) comme hypothèse simple ; monétisation réelle reportée.  
- **Pourquoi :** Consigne autorise l’exemple ; pas de chiffres pertes/marges fournis ; préférence projet pour une **solution simple**. Calibrage possible plus tard via perte moyenne / marge (voir `choix_techniques.txt` §6.bis).  
- **Conséquences :** Logger `fn_cost`/`fp_cost` dans MLflow ; **optimiser le seuil** sous ce coût (pas rester à 0.5).

---

## D-012 — Prochaines décisions à trancher (backlog)

| ID | Sujet | Options envisagées | Statut |
|----|--------|-------------------|--------|
| D-012a | Validation robuste | StratifiedKFold | **Tranché → D-013** |
| D-012b | Champion modèle | RF / XGBoost / LGBM / LR | **Tranché → D-013** (LightGBM) |
| D-012c | Hyperparams | Optuna vs GridSearchCV (simple) | **Tranché → D-015** (Optuna) |
| D-012d | Seuil | Balayage 0.05–0.95, min 10×FN+FP | **Tranché → D-014** |
| D-012e | Explicabilité | SHAP global + local | **Tranché → D-016** |
| D-012f | Registry + serving | MLflow Registry + serve | **Tranché → D-017** |

---

## D-013 — StratifiedKFold + multi-modèles

- **Statut :** Accepté  
- **Contexte :** Consigne étape 3 — CV obligatoire ; plusieurs familles ; éval split unique insuffisante.  
- **Décision :**
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sur `train.parquet`
  - Modèles : LogisticRegression, RandomForest, LightGBM, XGBoost (hyperparams baseline)
  - Proba **OOF** pour AUC CV + coût ; refit train → métriques holdout `valid`
  - Code : `src/modeling.py` + `src/run_cv_baselines.py` ; expérience MLflow `credit-scoring-fe_v1`, tag `stage=cv_threshold`
- **Pourquoi :** Stratification = distribution TARGET stable ; OOF = estimation non biaisée sans « user » le valid pour la CV.  
- **Résultats (vérifiés) :**

  | Modèle | CV AUC mean | Valid AUC | Valid coût@0.5 |
  |--------|-------------|-----------|----------------|
  | LightGBM | 0.779 | 0.784 | **30013** |
  | XGBoost | 0.774 | 0.780 | 30421 |
  | LogReg | 0.768 | 0.770 | 31666 |
  | RandomForest | 0.757 | 0.762 | 32324 |

- **Champion provisoire :** LightGBM (meilleur AUC CV + meilleur coût valid@0.5).  
- **Conséquences :** Optuna/GridSearch ciblé sur LightGBM ; RF baseline utile pour la comparaison mais non retenu.

---

## D-014 — Optimisation du seuil métier (coût 10:1)

- **Statut :** Accepté  
- **Contexte :** Consigne — ne pas garder 0.5 sans justification ; minimiser `10*FN + 1*FP`.  
- **Décision :**
  1. Balayage seuils 0.05→0.95 (pas 0.01) sur proba **OOF** (comparaison équitable entre modèles)
  2. Pour le champion LightGBM : re-optimiser le seuil sur le **holdout valid** (`src/run_threshold_on_valid.py`)
- **Résultat clé (vérifié) :** sur valid LightGBM, **seuil\* = 0.5** minimise le coût (30013). Voisins 0.49 / 0.51 sont légèrement pires.  
- **Observation :** les seuils OOF (ex. LGBM 0.48) dégradent un peu le coût valid (30267) — courbe plate près de 0.5 ; le transfert OOF→valid n’est pas parfait.  
- **Pourquoi c’est conforme :** le seuil a été **optimisé** ; 0.5 n’est plus un défaut non testé, c’est le minimum empirique sous 10:1 pour ce modèle.  
- **Conséquences :** logger `threshold_opt`, courbes CSV/PNG dans MLflow (`artifact_path=threshold`) ; après Optuna, **refaire** le balayage seuil (les probas changeront).

---

## D-015 — Optuna sur LightGBM (champion)

- **Statut :** Accepté  
- **Contexte :** Consigne étape 4 — optimiser les hyperparams (GridSearchCV ou équivalent). Champion CV = LightGBM (D-013).  
- **Décision :**
  - **Optuna** (TPE, 25 trials) plutôt que GridSearchCV : espace continu (lr, lambda) plus naturel ; déjà dans les deps.
  - Objectif Optuna = **maximiser AUC CV** (3 folds) — métrique de ranking stable, indépendante du seuil.
  - Puis **seuil\*** re-optimisé sur valid sous coût 10:1 (comme D-014).
  - Éval finale : StratifiedKFold **5** folds OOF + holdout valid.
  - Script : `src/run_optuna_lgbm.py` ; run parent MLflow `lightgbm_optuna_fe_v1` + trials nestés.
- **Pourquoi pas GridSearch :** grille dense sur 7 params = explosion combinatoire ; Optuna explore mieux avec un budget fixe de trials.  
- **Pourquoi pas optimiser directement le coût métier dans Optuna :** le coût dépend du seuil ; couplage HPO+seuil rend l’objectif bruité. Séparer HPO (AUC) / seuil (coût) est plus lisible.  
- **Résultats (vérifiés) :**
  - Best trial #21 — search AUC ≈ **0.7805** ; final CV AUC 5-fold ≈ **0.7817**
  - Valid AUC ≈ **0.7858** (baseline CV LGBM ≈ 0.784)
  - Valid coût@0.5 ≈ 30073 ; **coût@seuil\*=0.52 ≈ 29962** (baseline 30013 @0.5)
  - Params : `n_estimators=550`, `learning_rate≈0.021`, `num_leaves=47`, `min_child_samples=115`, `subsample≈0.70`, `colsample_bytree≈0.67`, `reg_lambda≈1.05`
- **Conséquences :** champion courant = LightGBM Optuna + seuil 0.52 ; artefacts `reports/optuna_lgbm_best.json` ; suite = SHAP puis Registry/serving.

---

## D-016 — SHAP global + local (explicabilité)

- **Statut :** Accepté  
- **Contexte :** Mission + notes : importance globale et locale pour le chargé d’études.  
- **Décision :** `shap.TreeExplainer` sur LightGBM Optuna ; global sur 1500 lignes valid ; 3 waterfalls locaux (refus / accord / proche seuil 0.52). Script `src/run_shap_lgbm.py`.  
- **Pourquoi SHAP :** standard pour arbres, valeurs additives interprétables (contribution ± au score).  
- **Résultats :** top global = EXT_SOURCE_2/3/1, CREDIT_GOODS_RATIO, ORGANIZATION_TYPE, CODE_GENDER, LATE_RATE…  
- **Point d’attention :** `CODE_GENDER` influent → documenter le risque équité ; pas d’exclusion automatique sans arbitrage métier.  
- **Conséquences :** artefacts `reports/shap/` ; run MLflow `lightgbm_shap_fe_v1`.

---

## D-017 — Model Registry + serving

- **Statut :** Accepté  
- **Contexte :** Exigence Michaël (registry + tester serving).  
- **Décision :**
  - Modèle enregistré `credit-scoring-lgbm` (v2 Staging, alias `@champion`)
  - Wrapper `mlflow.pyfunc` (`CreditLGBMWrapper`) pour inférence robuste (catégories)
  - Serving testé : `POST /invocations` → probabilités ; seuil 0.52 appliqué côté appli
- **Pourquoi wrapper pyfunc :** le flavor LightGBM natif + schema MLflow cassait sur dtypes `category`.  
- **Conséquences :** `src/run_register_champion.py`, `reports/registry_champion.json` ; commande serve dans `docs/commandes_utiles.txt`.

---

## D-018 — API FastAPI de scoring (partie 2)

- **Statut :** Accepté (implémenté 2026-11-21)
- **Contexte :** Consigne partie 2 étape 2 — exposer le modèle via une API, charger le modèle une seule fois, valider les entrées, documenter (Swagger).
- **Décision :**
  - Framework **FastAPI** (Swagger natif, typage Pydantic, adapté CI/CD) plutôt que Gradio.
  - Réponse = `{proba_default, threshold, decision, latency_ms}` avec seuil\* **0.52** appliqué côté API (`proba >= 0.52` → REFUS).
  - Booster LightGBM chargé dans le **lifespan** FastAPI (`api/model_runtime.py`) — une fois au démarrage.
  - Schéma features versionné : `reports/api_feature_schema.json` (+ échantillon `reports/api_sample_client.json`).
  - Validation : champs manquants, types incorrects, plages invalides (âge < 0, revenu ≤ 0, …) → HTTP 422.
  - Logs JSONL locaux : `logs/predictions.jsonl` (base pour étape 3 drift).
  - Conteneur : `Dockerfile` + `.dockerignore` (sans `data/` ni `mlruns/`).
  - Image **slim** via `requirements-api.txt` (FastAPI/LightGBM/pandas seulement) — pas tout `pyproject.toml` (sinon téléchargement XGBoost/MLflow/CUDA énorme et fragile).
- **Pourquoi pas Gradio :** moins adapté aux tests Pytest / pipeline CI ; FastAPI = contrat HTTP clair pour un chargé d’études / SI.
- **Déploiement cloud :** **non obligatoire**. Consigne étape 2 : « environnement cible (**simulé localement ou réel**) ». Docker local = déploiement simulé conforme. HF/Heroku/Cloud Run = outils optionnels listés.
- **Validation :**
  - `uv run pytest tests/test_api.py` → 7 passed.
  - Docker slim OK (2026-11-21) : `docker build -t credit-scoring-api` + run `-p 8001:8000` → `/health` OK, `/predict` → proba≈0.291 / ACCORD.
- **Conséquences :** CI GitHub Actions ajoutée (D-020) ; suite = drift Evidently ; opti.

---

## D-020 — GitHub Actions CI/CD (partie 2 étape 2)

- **Statut :** Accepté (2026-11-22)
- **Contexte :** Consigne — pipeline qui teste, build l’image Docker, déploie (local simulé OK).
- **Décision :** workflow `.github/workflows/ci.yml` sur push/PR (`EDA`, `main`) :
  1. **test** : `uv sync` + `pytest tests/test_api.py`
  2. **build** : `docker build` (image slim) + smoke `/health` + `/model/info`
  3. **deploy-simulated** : charge l’artifact image (`docker load`) — pas de cloud
- **Pourquoi pas de push registry cloud :** consigne autorise « simulé localement » ; pas de secrets GHCR requis pour le livrable.
- **Conséquences :** run visible dans l’onglet Actions du repo ; suite = drift (étape 3).


---

## D-019 — Dépôt Git « lean » (partie 2 étape 1, partiel)

- **Statut :** Accepté (2026-11-21)
- **Contexte :** Push GitHub échoué (HTTP 408) — pack ~860 Mo (`data/raw`, `mlruns`, …).
- **Décision :**
  - Enrichir `.gitignore` : `data/`, `mlruns/`, `mlflow.db`, `logs/`, archives zip/rar.
  - Réécrire l’historique local (reset sur commit initial + un commit propre).
  - Pousser la branche `EDA` : https://github.com/Mathieurgrd/AIE_project7/tree/EDA
- **Pourquoi :** GitHub coupe les pushes trop lourds ; les CSV Home Credit se régénèrent / se retéléchargent ; MLflow se rejoue.
- **Reste ouvert :** dépôt encore partiellement « à ranger » (PR vers `main`, README d’exécution, rendre le repo clairement public / default branch).

---

## Index rapide

| Décision | Thème |
|----------|--------|
| D-001 | Outillage env |
| D-002 | Archi notebooks / src |
| D-003 | FE fe_v1 |
| D-004 | Anti-leakage |
| D-005 | Clean EDA |
| D-006 | Split |
| D-007 | Kernels |
| D-008 | MLflow SQLite |
| D-009 | Baselines |
| D-010 | Class weight |
| D-011 | Coût 10:1 |
| D-012 | Backlog |
| D-013 | StratifiedKFold + multi-modèles |
| D-014 | Seuil métier optimisé |
| D-015 | Optuna LightGBM |
| D-016 | SHAP explicabilité |
| D-017 | Registry + serving |
| D-018 | API FastAPI + Docker |
| D-019 | Git lean (exclure data/mlruns) |
| D-020 | GitHub Actions CI/CD |
