# Étude drift & anomalies ops (PoC local)

**Source données courantes :** simulation valid (n=500, inject_drift=True)

## Data drift
- Référence (train sample) : n=2000
- Courant : n=500
- Features comparées : EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AGE_YEARS, EMPLOYMENT_YEARS, CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_GOODS_RATIO, LATE_RATE, CNT_CHILDREN
- Colonnes driftées : 6.0 (share=0.46153846153846156)
- Features driftées : ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'LATE_RATE']
- Rapport HTML : `reports/drift/drift_report.html`

## Anomalies opérationnelles (logs API)
- Appels : 8
- OK : 3 | erreurs/validation : 5
- Taux d'erreur : 0.625
- Latence mean / p95 / max (ms) : 62.75797494163271 / 75.57213966792915 / 76.6824996098876
- Alerte taux d'erreur > 5 % : True
- Alerte latence p95 > 500 ms : False

## Points de vigilance
- Le drift nécessite une **référence stable** (ici : train fe_v1).
- Un drift détecté ≠ redeploy automatique : investiguer cause (population, bug FE, saisonnalité).
- Logs complets (features) : utiles au monitoring, attention RGPD en vrai prod (minimisation / anonymisation).
- Si `--inject-drift` : le signal est **artificiel** (démo pédagogique).
