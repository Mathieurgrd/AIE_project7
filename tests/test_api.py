"""Tests critiques API (consigne partie 2 étape 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "reports" / "api_sample_client.json"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_features() -> dict:
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return payload["features"]


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client: TestClient):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == 218
    assert body["threshold_opt_valid"] == 0.52


def test_predict_ok(client: TestClient, sample_features: dict):
    r = client.post("/predict", json={"features": sample_features})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["proba_default"] <= 1.0
    assert body["threshold"] == 0.52
    assert body["decision"] in {"ACCORD", "REFUS"}
    assert body["latency_ms"] >= 0


def test_predict_missing_features(client: TestClient):
    r = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": 100000}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(err.get("code") == "missing_features" for err in detail)


def test_predict_invalid_type(client: TestClient, sample_features: dict):
    bad = dict(sample_features)
    bad["AMT_CREDIT"] = "pas-un-nombre"
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(err.get("code") == "invalid_type" for err in detail)


def test_predict_out_of_range_age(client: TestClient, sample_features: dict):
    bad = dict(sample_features)
    bad["AGE_YEARS"] = -5
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(err.get("code") == "out_of_range" for err in detail)


def test_predict_income_zero(client: TestClient, sample_features: dict):
    bad = dict(sample_features)
    bad["AMT_INCOME_TOTAL"] = 0
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422
    assert any(err.get("code") == "out_of_range" for err in r.json()["detail"])
