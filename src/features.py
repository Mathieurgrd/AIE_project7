"""
Feature engineering fe_v1.

Rôle : transformer les tables Home Credit brutes en 1 ligne = 1 SK_ID_CURR
       (demande / client), avec aggs d'historique crédit / paiement.

Important anti-leakage :
  Ici = transformations DÉTERMINISTES seulement (ratios, groupby, flags).
  Pas d'imputation / OHE / scaler « appris » — ça se fait après le split
  dans les pipelines modèles (voir modeling.make_sklearn_preprocess).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FE_VERSION = "fe_v1"
# Valeur sentinelle Home Credit = « pas d'emploi » (pas une vraie ancienneté).
DAYS_EMPLOYED_SENTINEL = 365243


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division robuste : 0 / inf → NaN (les arbres gèrent les NA)."""
    out = num / den.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# application_* : fiche demande
# ---------------------------------------------------------------------------

def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage + features simples (âge, charge crédit/revenu, flag logement)."""
    df = df.copy()

    # Sentinelle → NaN + flag explicite (l'absence d'info peut être un signal).
    df["DAYS_EMPLOYED_ANOMALY"] = (
        df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    ).astype(np.int8)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
        {DAYS_EMPLOYED_SENTINEL: np.nan}
    )

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]) / 365.25

    # Ratios = charge relative (souvent plus parlants que les montants bruts).
    df["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_GOODS_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])

    # NA logement souvent en bloc → l'absence elle-même est informative.
    if "COMMONAREA_AVG" in df.columns:
        df["HOUSING_MISSING"] = df["COMMONAREA_AVG"].isna().astype(np.int8)
    else:
        df["HOUSING_MISSING"] = 0

    return df


def _prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Préfixe les colonnes (sauf SK_ID_CURR) pour savoir d'où vient la feature."""
    df = df.copy()
    df.columns = [
        c if c == "SK_ID_CURR" else f"{prefix}{c}" for c in df.columns
    ]
    return df


# ---------------------------------------------------------------------------
# Tables 1→N : on agrège jusqu'à 1 ligne / client
# ---------------------------------------------------------------------------

def agg_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Historique crédits hors Home Credit → stats + comptes Active/Closed/…"""
    b = bureau.copy()
    aggs = {
        "SK_ID_BUREAU": ["count"],
        "DAYS_CREDIT": ["mean", "min", "max"],
        "CREDIT_DAY_OVERDUE": ["mean", "max"],
        "DAYS_CREDIT_ENDDATE": ["mean", "min"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
        "CNT_CREDIT_PROLONG": ["sum", "mean"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_LIMIT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean", "max"],
        "DAYS_CREDIT_UPDATE": ["mean", "min"],
        "AMT_ANNUITY": ["mean", "sum"],
    }
    aggs = {k: v for k, v in aggs.items() if k in b.columns}
    out = b.groupby("SK_ID_CURR").agg(aggs)
    out.columns = ["_".join(c).upper() for c in out.columns]
    out = out.reset_index()

    # One-hot comptage du statut Active / Closed / …
    status = (
        b.groupby(["SK_ID_CURR", "CREDIT_ACTIVE"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    status.columns = [
        c if c == "SK_ID_CURR" else f"CREDIT_ACTIVE_{c}".upper().replace(" ", "_")
        for c in status.columns
    ]
    out = out.merge(status, on="SK_ID_CURR", how="left")
    return _prefix(out, "BUREAU_")


def agg_bureau_balance(
    bureau_balance: pd.DataFrame, bureau: pd.DataFrame
) -> pd.DataFrame:
    """
    Historique mensuel des crédits bureau.
    Chemin : mois → crédit (SK_ID_BUREAU) → client (SK_ID_CURR).
    """
    bb = bureau_balance.copy()
    # STATUS '1'..'5' = en retard ; 'C' fermé ; 'X' inconnu ; '0' OK.
    bb["IS_DPD"] = bb["STATUS"].isin(list("12345")).astype(np.int8)
    bb["IS_CLOSED"] = (bb["STATUS"] == "C").astype(np.int8)
    bb["IS_UNKNOWN"] = (bb["STATUS"] == "X").astype(np.int8)
    bb["IS_OK"] = (bb["STATUS"] == "0").astype(np.int8)

    per_bureau = bb.groupby("SK_ID_BUREAU").agg(
        MONTHS=("MONTHS_BALANCE", "count"),
        DPD_MONTHS=("IS_DPD", "sum"),
        CLOSED_MONTHS=("IS_CLOSED", "sum"),
        UNKNOWN_MONTHS=("IS_UNKNOWN", "sum"),
        OK_MONTHS=("IS_OK", "sum"),
        MONTHS_MIN=("MONTHS_BALANCE", "min"),
    )
    per_bureau["DPD_RATE"] = per_bureau["DPD_MONTHS"] / per_bureau["MONTHS"]

    link = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].drop_duplicates()
    merged = link.merge(per_bureau, on="SK_ID_BUREAU", how="inner")
    out = merged.groupby("SK_ID_CURR").agg(
        BB_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BB_MONTHS_SUM=("MONTHS", "sum"),
        BB_DPD_RATE_MEAN=("DPD_RATE", "mean"),
        BB_DPD_RATE_MAX=("DPD_RATE", "max"),
        BB_DPD_MONTHS_SUM=("DPD_MONTHS", "sum"),
        BB_MONTHS_MIN_MEAN=("MONTHS_MIN", "mean"),
    )
    return out.reset_index()


def agg_previous(previous: pd.DataFrame) -> pd.DataFrame:
    """Anciennes demandes Home Credit + taux Approved / Refused / …"""
    p = previous.copy()
    aggs = {
        "SK_ID_PREV": ["count"],
        "AMT_ANNUITY": ["mean", "max", "min"],
        "AMT_APPLICATION": ["mean", "max", "min"],
        "AMT_CREDIT": ["mean", "max", "min", "sum"],
        "AMT_DOWN_PAYMENT": ["mean", "sum"],
        "AMT_GOODS_PRICE": ["mean", "max"],
        "DAYS_DECISION": ["mean", "min", "max"],
        "CNT_PAYMENT": ["mean", "max"],
        "RATE_DOWN_PAYMENT": ["mean"],
    }
    aggs = {k: v for k, v in aggs.items() if k in p.columns}
    out = p.groupby("SK_ID_CURR").agg(aggs)
    out.columns = ["_".join(c).upper() for c in out.columns]
    out = out.reset_index()

    status = (
        p.groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    status.columns = [
        c
        if c == "SK_ID_CURR"
        else f"STATUS_{c}".upper().replace(" ", "_")
        for c in status.columns
    ]
    out = out.merge(status, on="SK_ID_CURR", how="left")
    # Ratios de statut (ex. part de dossiers refusés dans le passé).
    n = out["SK_ID_PREV_COUNT"].replace({0: np.nan})
    for col in out.columns:
        if col.startswith("STATUS_"):
            out[f"RATE_{col}"] = out[col] / n
    return _prefix(out, "PREV_")


def agg_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Échéances : écarts de paiement + retards (fort signal de défaut)."""
    i = inst.copy()
    i["PAYMENT_DIFF"] = i["AMT_PAYMENT"] - i["AMT_INSTALMENT"]
    i["DAYS_LATE"] = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]
    i["IS_LATE"] = (i["DAYS_LATE"] > 0).astype(np.int8)

    out = i.groupby("SK_ID_CURR").agg(
        INST_COUNT=("SK_ID_PREV", "count"),
        PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
        PAYMENT_DIFF_SUM=("PAYMENT_DIFF", "sum"),
        PAYMENT_DIFF_MIN=("PAYMENT_DIFF", "min"),
        DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        DAYS_LATE_MAX=("DAYS_LATE", "max"),
        LATE_RATE=("IS_LATE", "mean"),
        AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
    )
    return out.reset_index()


def agg_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Soldes POS/cash : retards DPD et échéances restantes."""
    out = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "count"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
    )
    return out.reset_index()


def agg_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Cartes : utilisation du plafond + retards."""
    c = cc.copy()
    c["UTILIZATION"] = _safe_div(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    out = c.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "count"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION", "max"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_PAYMENT_MEAN=("AMT_PAYMENT_CURRENT", "mean"),
        CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
    )
    return out.reset_index()


def merge_features(base: pd.DataFrame, *aggs: pd.DataFrame) -> pd.DataFrame:
    """Left-join des aggs sur application. Garde-fou : pas d'explosion de lignes."""
    out = base.copy()
    n0 = len(out)
    for a in aggs:
        if a is None or len(a) == 0:
            continue
        before_cols = set(out.columns)
        out = out.merge(a, on="SK_ID_CURR", how="left")
        if len(out) != n0:
            raise ValueError(
                f"Explosion de lignes au merge: {n0} → {len(out)} "
                f"(vérifier unicité SK_ID_CURR dans l'agg)"
            )
        new_cols = [c for c in out.columns if c not in before_cols]
        _ = new_cols
    return out


def build_full_tables(
    raw_dir: Path,
    include_bureau_balance: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Orchestrateur FE : charge raw → clean + aggs → merge.
    Retourne (train_full, test_full, log) prêts à splitter / sauver.
    """
    raw_dir = Path(raw_dir)
    log: dict = {"fe_version": FE_VERSION, "steps": []}

    print("Chargement application_train/test…")
    train = pd.read_csv(raw_dir / "application_train.csv")
    test = pd.read_csv(raw_dir / "application_test.csv")
    log["steps"].append(f"application_train shape={train.shape}")

    train = clean_application(train)
    test = clean_application(test)

    aggs: list[pd.DataFrame] = []

    print("Agrégation bureau…")
    bureau = pd.read_csv(raw_dir / "bureau.csv")
    bureau_agg = agg_bureau(bureau)
    aggs.append(bureau_agg)
    log["steps"].append(f"bureau_agg shape={bureau_agg.shape}")

    if include_bureau_balance:
        try:
            print("Agrégation bureau_balance (peut être long)…")
            bb = pd.read_csv(raw_dir / "bureau_balance.csv")
            bb_agg = agg_bureau_balance(bb, bureau)
            aggs.append(bb_agg)
            log["steps"].append(f"bureau_balance_agg shape={bb_agg.shape}")
            log["bureau_balance"] = "included"
            del bb
        except Exception as exc:  # noqa: BLE001
            log["bureau_balance"] = f"skipped: {exc}"
            print("WARN bureau_balance skippé:", exc)
    else:
        log["bureau_balance"] = "disabled"

    del bureau

    print("Agrégation previous_application…")
    previous = pd.read_csv(raw_dir / "previous_application.csv")
    prev_agg = agg_previous(previous)
    aggs.append(prev_agg)
    log["steps"].append(f"previous_agg shape={prev_agg.shape}")
    del previous

    print("Agrégation installments_payments…")
    inst = pd.read_csv(raw_dir / "installments_payments.csv")
    inst_agg = agg_installments(inst)
    aggs.append(inst_agg)
    log["steps"].append(f"installments_agg shape={inst_agg.shape}")
    del inst

    print("Agrégation POS_CASH_balance…")
    pos = pd.read_csv(raw_dir / "POS_CASH_balance.csv")
    pos_agg = agg_pos_cash(pos)
    aggs.append(pos_agg)
    log["steps"].append(f"pos_agg shape={pos_agg.shape}")
    del pos

    print("Agrégation credit_card_balance…")
    cc = pd.read_csv(raw_dir / "credit_card_balance.csv")
    cc_agg = agg_credit_card(cc)
    aggs.append(cc_agg)
    log["steps"].append(f"credit_card_agg shape={cc_agg.shape}")
    del cc

    print("Merges…")
    train_full = merge_features(train, *aggs)
    test_full = merge_features(test, *aggs)
    log["steps"].append(f"train_full shape={train_full.shape}")
    log["steps"].append(f"test_full shape={test_full.shape}")
    log["n_features_train"] = train_full.shape[1] - 2  # hors SK_ID + TARGET

    return train_full, test_full, log
