# Rapport d'optimisation d'inférence (étape 4)

## Contexte
Après déploiement API + logs, la latence unitaire ~50 ms était trop élevée
pour du scoring ligne-à-ligne. Objectif : réduire le temps d'inférence
sans régression de précision (consigne partie 2 étape 4).

## Goulot identifié (cProfile)
- Hot path = construction DataFrame pandas + casts colonnes + `_data_from_pandas`.
- Le booster LightGBM lui-même n'est pas le coût dominant en single-row.
- Détail : `reports/optimization/cprofile_baseline.txt`

## Stratégies testées
1. **Optimisation de code** : encodage numpy des catégorielles via `pandas_categorical` du modèle.
2. **ONNX Runtime** : export `onnxmltools.convert_lightgbm` (zipmap=False) + inference CPU.
3. (Non retenu) re-entraîner sans cats natives — risque métier inutile vu le succès ONNX.

## Résultats latence (single-row, sample client)
- Baseline LightGBM+pandas : **52.108 ms**
- ONNX Runtime + numpy : **0.246 ms**
- Speedup : **×211.8**
- |Δproba| sample : 2.15e-08

## Fidélité (holdout valid)
- AUC LightGBM (pandas) : 0.769024
- AUC ONNX : 0.769024
- max |Δproba| : 2.62e-03
- mean |Δproba| : 2.31e-06
- n lignes valid : 2000

## Configuration finale
- Runtime API défaut : **ONNX Runtime** (`models/champion_lgbm_optuna.onnx`)
- Maps catégories : `models/champion_lgbm_cat_maps.json`
- Fallback : `SCORING_RUNTIME=lgbm` (fichier `.txt` + pandas)
- Image Docker : `onnxruntime` (plus besoin de LightGBM au runtime)

## Décision
Adopter ONNX Runtime dans l'API (défaut). Fallback LightGBM+pandas via SCORING_RUNTIME=lgbm.
