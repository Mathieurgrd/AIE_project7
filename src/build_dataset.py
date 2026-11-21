"""
Construit les datasets interim / processed pour fe_v1.

Entrée  : data/raw/*.csv
Sortie  :
  - data/interim/{train,test}_full.parquet  (FE complet, avant split)
  - data/processed/{train,valid,test}.parquet + meta.json
  - append d'un journal dans choix_techniques.txt

Lancer : uv run python -m src.build_dataset
         uv run python -m src.build_dataset --no-bb   # sans bureau_balance
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FE_VERSION, build_full_tables  # noqa: E402

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
CHOIX = ROOT / "choix_techniques.txt"

RANDOM_STATE = 42
VALID_SIZE = 0.2  # 80 % train / 20 % valid, stratifié sur TARGET


def main(include_bureau_balance: bool = True) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1) FE multi-tables (1 ligne = 1 SK_ID_CURR)
    train_full, test_full, log = build_full_tables(
        RAW, include_bureau_balance=include_bureau_balance
    )

    # 2) Interim = snapshot FE avant split (rejouable / debug)
    print("Écriture interim…")
    train_path = INTERIM / "train_full.parquet"
    test_path = INTERIM / "test_full.parquet"
    train_full.to_parquet(train_path, index=False)
    test_full.to_parquet(test_path, index=False)

    # 3) Split stratifié : même taux de défaut train/valid (~8 %)
    y = train_full["TARGET"]
    train_df, valid_df = train_test_split(
        train_full,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 4) Processed = ce que les scripts modèles chargent
    print("Écriture processed…")
    train_out = PROCESSED / "train.parquet"
    valid_out = PROCESSED / "valid.parquet"
    test_out = PROCESSED / "test.parquet"
    train_df.to_parquet(train_out, index=False)
    valid_df.to_parquet(valid_out, index=False)
    test_full.to_parquet(test_out, index=False)  # pas de TARGET (Kaggle)

    # 5) meta.json = traçabilité (version FE, tailles, seed, chemins)
    meta = {
        "fe_version": FE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "valid_size": VALID_SIZE,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_full)),
        "train_cols": int(train_df.shape[1]),
        "target_rate_train": float(train_df["TARGET"].mean()),
        "target_rate_valid": float(valid_df["TARGET"].mean()),
        "include_bureau_balance": include_bureau_balance,
        "log": log,
        "files": {
            "interim_train": str(train_path.relative_to(ROOT)),
            "interim_test": str(test_path.relative_to(ROOT)),
            "processed_train": str(train_out.relative_to(ROOT)),
            "processed_valid": str(valid_out.relative_to(ROOT)),
            "processed_test": str(test_out.relative_to(ROOT)),
        },
    }
    meta_path = PROCESSED / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 6) Append journal humain dans choix_techniques.txt
    journal = f"""
--- RUN {meta['created_at_utc']} ---
fe_version={FE_VERSION}
include_bureau_balance={include_bureau_balance}
train_full={train_full.shape} -> split train={train_df.shape} valid={valid_df.shape}
test_full={test_full.shape}
target_rate_train={meta['target_rate_train']:.4f} valid={meta['target_rate_valid']:.4f}
bureau_balance={log.get('bureau_balance')}
steps={log.get('steps')}
artefacts={meta['files']}
status=OK
"""
    with CHOIX.open("a", encoding="utf-8") as f:
        f.write(journal)

    print("OK — meta:", meta_path)
    print(journal)


if __name__ == "__main__":
    include_bb = "--no-bb" not in sys.argv
    main(include_bureau_balance=include_bb)
