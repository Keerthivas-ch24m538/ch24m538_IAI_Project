import json
import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.app import app

client = TestClient(app)

def get_sample_payload():
    # Load features from your saved file for a valid minimal input
    with open("reports/features.json") as f:
        features = json.load(f)["features"]
    # Minimal valid dict with all required numeric keys set to 1.0
    record = {k: 1.0 for k in features}
    return {"records": [record]}

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_predict_valid():
    payload = get_sample_payload()
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    out = resp.json()
    assert "preds" in out and "probs" in out
    assert isinstance(out["preds"], list)
    assert isinstance(out["probs"], list)

def test_predict_missing_key():
    payload = get_sample_payload()
    # Remove one key
    key = list(payload["records"][0].keys())[0]
    del payload["records"][0][key]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert data["detail"]["error"] == "feature_mismatch"
    assert key in data["detail"]["missing"]

def test_predict_extra_key():
    payload = get_sample_payload()
    payload["records"][0]["not_a_feature"] = 123
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert data["detail"]["error"] == "feature_mismatch"
    assert "not_a_feature" in data["detail"]["extra"]

