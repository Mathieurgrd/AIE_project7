"""Logging structuré JSONL des prédictions (étape 3 consigne — base locale)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from api.config import LOG_DIR, PREDICTION_LOG


def append_prediction_log(record: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with PREDICTION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
