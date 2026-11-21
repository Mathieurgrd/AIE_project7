# Optuna — résumé pédagogique

Document de compréhension (pas un journal de décision).  
Usage projet : voir **D-015** dans `docs/Decision.md` et `src/run_optuna_lgbm.py`.

---

## En une phrase

**Optuna** est une bibliothèque Python d’**optimisation d’hyperparamètres** : elle cherche automatiquement les meilleurs réglages d’un modèle (learning rate, profondeur, etc.) pour maximiser (ou minimiser) une métrique que tu définis.

Analogie logiciel : comme un **testeur A/B automatisé** sur les knobs d’un algorithme — tu fixes l’espace de recherche et le score, Optuna propose des essais, mesure, et se concentre sur les zones prometteuses.

---

## Problème qu’elle résout

Un modèle ML a souvent des **hyperparamètres** (réglages choisis *avant* l’apprentissage, pas appris directement sur les données) :

| Exemple LightGBM | Rôle |
|------------------|------|
| `learning_rate` | Pas d’apprentissage |
| `num_leaves` | Complexité des arbres |
| `n_estimators` | Nombre d’arbres |
| `reg_lambda` | Régularisation |

Les tester à la main est long. Une **grille** (`GridSearchCV`) teste toutes les combinaisons → explosion combinatoire dès que l’espace est grand ou continu.

Optuna explore un **budget fixe d’essais** (*trials*) de façon plus intelligente qu’une grille naïve.

---

## Vocabulaire clé

| Terme | Sens |
|-------|------|
| **Study** | L’étude d’optimisation complète (objectif + historique des essais) |
| **Trial** | Un essai = un jeu d’hyperparams + le score obtenu |
| **Objective** | Fonction à optimiser (ex. maximiser l’AUC en CV) |
| **Suggest** | Tirage d’une valeur dans l’espace (`suggest_float`, `suggest_int`, …) |
| **Sampler** | Stratégie de proposition des prochains essais (ex. **TPE**) |
| **Pruning** | (Optionnel) Arrêt anticipé d’un essai clairement mauvais |

---

## Comment ça marche (schéma)

```
1. Tu définis objective(trial):
      params = trial.suggest_...(…)
      score  = entraîner + évaluer (souvent en CV)
      return score

2. Optuna crée une Study (direction = maximize ou minimize)

3. Pour n_trials fois :
      - le Sampler propose des params (au début ≈ exploration,
        puis davantage autour des bons scores)
      - objective est exécutée
      - le résultat est enregistré

4. study.best_params / study.best_value
```

Le sampler **TPE** (*Tree-structured Parzen Estimator*) modélise « quels params donnent de bons scores » vs « de mauvais », puis tire plutôt du côté prometteur. Ce n’est pas de la magie : avec trop peu de trials, le gain peut être faible.

---

## Optuna vs GridSearchCV / RandomizedSearchCV

| Approche | Idée | Limite |
|----------|------|--------|
| **GridSearchCV** | Toutes les combos d’une grille | Coût explosif ; mal adapté aux floats fins |
| **RandomizedSearchCV** | Tirages aléatoires dans l’espace | Simple, mais n’exploite pas l’historique |
| **Optuna** | Tirages guidés (TPE, etc.) + API souple | Dépend du budget (`n_trials`) et de la qualité de l’objectif |

Sur ce projet : Optuna choisi (D-015) car l’espace LightGBM est surtout **continu** et on voulait un budget fixe (~25 trials), pas une grille dense.

---

## Lien avec la cross-validation

Optuna **ne remplace pas** la CV : elle l’utilise souvent *à l’intérieur* de l’objective.

- Sans CV : risque de choisir des hyperparams qui surapprennent un seul split.
- Avec `StratifiedKFold` dans l’objective : le score (ex. AUC moyenne) est plus robuste.

Ici : recherche avec **3 folds** (plus rapide), puis confirmation du meilleur jeu avec **5 folds** + holdout `valid`.

---

## Ce qu’Optuna n’est pas

- **Pas** un modèle de ML (elle ne prédit pas le défaut client).
- **Pas** un remplace­ment de MLflow (MLflow *tracke* ; Optuna *cherche*).
- **Pas** une garantie de meilleur score monde : si l’espace est mal choisi, ou `n_trials` trop petit, le gain peut être marginal.
- **Pas** l’optimisation du **seuil** métier : le seuil (0/1 à partir de la proba) se traite à part (chez nous : balayage coût FN/FP après les meilleurs params).

---

## Usage dans ce projet

| Élément | Choix |
|---------|--------|
| Modèle | LightGBM (champion CV) |
| Objectif Optuna | Maximiser **AUC CV** |
| Sampler | TPE, seed 42 |
| Budget | 25 trials |
| Après Optuna | Seuil\* = argmin `10*FN + FP` sur `valid` |
| Script | `uv run python -m src.run_optuna_lgbm` |
| Artefact | `reports/optuna_lgbm_best.json` |
| Tracking | Run MLflow `lightgbm_optuna_fe_v1` (+ trials nestés) |

Pourquoi optimiser l’**AUC** et pas directement le **coût métier** dans Optuna ?  
Le coût dépend du seuil. Mélanger HPO + seuil dans un seul objectif rend le signal bruité et plus dur à interpréter. Séparation volontaire : Optuna = ranking (AUC) ; seuil = décision métier (D-014 / D-015).

---

## Mini exemple de structure de code

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    leaves = trial.suggest_int("num_leaves", 16, 64)
    # ... entraîner + CV ...
    return cv_auc_mean  # à maximiser

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)
print(study.best_params, study.best_value)
```

---

## Pour aller plus loin (hors scope immédiat)

- **Pruning** : couper tôt les trials médiocres (gain de temps).
- **Multi-objectifs** : ex. AUC vs temps d’inférence.
- **Intégrations** : `optuna-integration` avec LightGBM / sklearn, dashboards Optuna.
- Doc officielle : [https://optuna.org](https://optuna.org)
