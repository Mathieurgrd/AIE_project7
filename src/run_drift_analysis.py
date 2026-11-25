"""
Analyse de drift + anomalies opérationnelles (consigne partie 2 étape 3).

PoC 100 % local :
  - Référence = échantillon train fe_v1
  - Courant = features loguées par l'API (logs/predictions.jsonl)
    ou, si logs insuffisants : simulation depuis valid (+ option drift artificiel)

Sorties :
  reports/drift/drift_report.html
  reports/drift/ops_summary.json
  reports/drift/drift_summary.md

Usage :
  uv run python -m src.run_drift_analysis
  uv run python -m src.run_drift_analysis --simulate-n 500 --inject-drift
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LOG_PATH = ROOT / "logs" / "predictions.jsonl"
OUT_DIR = ROOT / "reports" / "drift"
SCHEMA_PATH = ROOT / "reports" / "api_feature_schema.json"
RANDOM_STATE = 42

# Features numériques les plus stables / informatives pour un rapport lisible.
# (Evidently sur 218 colonnes est lent et bruyant pour un PoC mentor.)
FOCUS_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "LATE_RATE",
    "CNT_CHILDREN",
]


def _load_schema_features() -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return list(schema["features"])


def _prepare_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    cols = [c for c in FOCUS_FEATURES if c in feature_names and c in df.columns]
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_reference(n: int = 2000) -> pd.DataFrame:
    train = pd.read_parquet(PROC / "train.parquet")
    feature_names = _load_schema_features()
    X = train.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return _prepare_frame(X.iloc[idx], feature_names)


def load_features_from_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (features_df, ops_df) depuis le JSONL API."""
    if not LOG_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    rows_feat: list[dict] = []
    rows_ops: list[dict] = []
    with LOG_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows_ops.append(
                {
                    "ts_utc": rec.get("ts_utc"),
                    "status": rec.get("status"),
                    "http_status": rec.get("http_status"),
                    "latency_ms": rec.get("latency_ms"),
                    "decision": rec.get("decision"),
                    "proba_default": rec.get("proba_default"),
                }
            )
            feats = rec.get("features")
            if isinstance(feats, dict) and rec.get("status") == "ok":
                rows_feat.append(feats)

    ops = pd.DataFrame(rows_ops)
    if not rows_feat:
        return pd.DataFrame(), ops
    feature_names = _load_schema_features()
    feat = _prepare_frame(pd.DataFrame(rows_feat), feature_names)
    return feat, ops


def simulate_current(n: int, inject_drift: bool) -> pd.DataFrame:
    """Simule un lot « production » à partir du holdout valid."""
    valid = pd.read_parquet(PROC / "valid.parquet")
    feature_names = _load_schema_features()
    X = valid.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    rng = np.random.default_rng(RANDOM_STATE + 1)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    cur = _prepare_frame(X.iloc[idx], feature_names)
    if inject_drift:
        # Drift artificiel pédagogique : baisse des scores externes + hausse LATE_RATE.
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            if col in cur.columns:
                cur[col] = (cur[col] * 0.7).clip(lower=0.0, upper=1.0)
        if "LATE_RATE" in cur.columns:
            cur["LATE_RATE"] = (cur["LATE_RATE"].fillna(0) + 0.15).clip(upper=1.0)
        if "AMT_CREDIT" in cur.columns:
            cur["AMT_CREDIT"] = cur["AMT_CREDIT"] * 1.25
    return cur


def analyze_ops(ops: pd.DataFrame) -> dict:
    if ops.empty:
        return {
            "n_calls": 0,
            "note": "Aucun log opérationnel — lancer l'API ou --simulate-n",
        }
    lat = pd.to_numeric(ops.get("latency_ms"), errors="coerce")
    status = ops.get("status", pd.Series(dtype=str))
    n = len(ops)
    n_ok = int((status == "ok").sum()) if status is not None else 0
    n_err = int(status.isin(["error", "validation_error"]).sum()) if status is not None else 0
    summary = {
        "n_calls": n,
        "n_ok": n_ok,
        "n_error_or_validation": n_err,
        "error_rate": float(n_err / n) if n else 0.0,
        "latency_ms_mean": float(lat.mean()) if lat.notna().any() else None,
        "latency_ms_p95": float(lat.quantile(0.95)) if lat.notna().any() else None,
        "latency_ms_max": float(lat.max()) if lat.notna().any() else None,
        "alert_high_error_rate": bool(n_err / n > 0.05) if n else False,
        "alert_high_latency_p95": bool(lat.quantile(0.95) > 500) if lat.notna().any() else False,
    }
    return summary


def run_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    html_path = OUT_DIR / "drift_report.html"
    json_path = OUT_DIR / "drift_report.json"
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    # Résumé lisible depuis le JSON Evidently 0.7.
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    drifted: list[str] = []
    share = None
    count_drifted = None
    col_pat = re.compile(
        r"ValueDrift\(column=(?P<col>[^,]+),method=(?P<method>[^,]+),threshold=(?P<th>[0-9.]+)\)"
    )
    for m in payload.get("metrics") or []:
        if not isinstance(m, dict):
            continue
        name = str(m.get("metric_name", ""))
        val = m.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(val, dict):
            share = val.get("share")
            count_drifted = val.get("count")
            continue
        match = col_pat.search(name)
        if not match or val is None:
            continue
        col = match.group("col")
        method = match.group("method")
        th = float(match.group("th"))
        score = float(val)
        # p-value : drift si score < seuil ; distance : drift si score > seuil.
        is_drift = score < th if "p_value" in method else score > th
        if is_drift:
            drifted.append(col)

    return {
        "html": str(html_path.relative_to(ROOT)).replace("\\", "/"),
        "json": str(json_path.relative_to(ROOT)).replace("\\", "/"),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "features_compared": list(reference.columns),
        "n_drifted_columns": count_drifted,
        "share_of_drifted_columns": share,
        "drifted_features_preview": drifted[:20],
    }


def write_markdown(drift_info: dict, ops_info: dict, source: str) -> Path:
    md_path = OUT_DIR / "drift_summary.md"
    lines = [
        "# Étude drift & anomalies ops (PoC local)",
        "",
        f"**Source données courantes :** {source}",
        "",
        "## Data drift",
        f"- Référence (train sample) : n={drift_info.get('n_reference')}",
        f"- Courant : n={drift_info.get('n_current')}",
        f"- Features comparées : {', '.join(drift_info.get('features_compared', []))}",
        f"- Colonnes driftées : {drift_info.get('n_drifted_columns')} "
        f"(share={drift_info.get('share_of_drifted_columns')})",
        f"- Features driftées : {drift_info.get('drifted_features_preview')}",
        f"- Rapport HTML : `{drift_info.get('html')}`",
        "",
        "## Anomalies opérationnelles (logs API)",
        f"- Appels : {ops_info.get('n_calls')}",
        f"- OK : {ops_info.get('n_ok')} | erreurs/validation : {ops_info.get('n_error_or_validation')}",
        f"- Taux d'erreur : {ops_info.get('error_rate')}",
        f"- Latence mean / p95 / max (ms) : {ops_info.get('latency_ms_mean')} / {ops_info.get('latency_ms_p95')} / {ops_info.get('latency_ms_max')}",
        f"- Alerte taux d'erreur > 5 % : {ops_info.get('alert_high_error_rate')}",
        f"- Alerte latence p95 > 500 ms : {ops_info.get('alert_high_latency_p95')}",
        "",
        "## Points de vigilance",
        "- Le drift nécessite une **référence stable** (ici : train fe_v1).",
        "- Un drift détecté ≠ redeploy automatique : investiguer cause (population, bug FE, saisonnalité).",
        "- Logs complets (features) : utiles au monitoring, attention RGPD en vrai prod (minimisation / anonymisation).",
        "- Si `--inject-drift` : le signal est **artificiel** (démo pédagogique).",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift + ops monitoring PoC")
    parser.add_argument("--simulate-n", type=int, default=0, help="Simuler N lignes depuis valid si logs vides/insuffisants")
    parser.add_argument("--inject-drift", action="store_true", help="Injecter un drift artificiel sur la simulation")
    parser.add_argument("--ref-n", type=int, default=2000, help="Taille échantillon référence train")
    parser.add_argument("--min-log-rows", type=int, default=50, help="Seuil mini de logs features pour skip simulate")
    args = parser.parse_args()

    if not (PROC / "train.parquet").exists():
        raise SystemExit(
            "data/processed/train.parquet manquant. "
            "Lancer : uv run python -m src.build_dataset"
        )

    print("Build référence (train)...", flush=True)
    reference = build_reference(n=args.ref_n)

    print("Lecture logs API...", flush=True)
    current_from_logs, ops = load_features_from_logs()
    source = "logs/predictions.jsonl"

    if len(current_from_logs) < args.min_log_rows:
        n_sim = args.simulate_n or 500
        print(
            f"Logs features insuffisants (n={len(current_from_logs)}) → simulation n={n_sim}",
            flush=True,
        )
        current = simulate_current(n=n_sim, inject_drift=args.inject_drift)
        source = f"simulation valid (n={n_sim}, inject_drift={args.inject_drift})"
    else:
        current = current_from_logs
        print(f"Utilisation logs features n={len(current)}", flush=True)

    # Aligner colonnes communes
    common = [c for c in reference.columns if c in current.columns]
    reference = reference[common]
    current = current[common]

    print("Evidently DataDriftPreset...", flush=True)
    drift_info = run_drift(reference, current)
    ops_info = analyze_ops(ops)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"source": source, "drift": drift_info, "ops": ops_info}
    (OUT_DIR / "ops_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = write_markdown(drift_info, ops_info, source)

    print(f"OK → {drift_info['html']}", flush=True)
    print(f"OK → {md.relative_to(ROOT)}", flush=True)
    print(f"OK → reports/drift/ops_summary.json", flush=True)


if __name__ == "__main__":
    main()
