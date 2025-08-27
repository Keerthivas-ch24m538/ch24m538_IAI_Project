# src/app.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .schemas import PredictRequest, PredictResponse

APP_TITLE = "Titanic API"
APP_VERSION = "0.2.0"
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# Global artifacts
MODEL_PATH = Path("models/current.joblib")
FEATURES_PATH = Path("reports/features.json")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")
try:
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f).get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("features.json missing or invalid 'features' list")
except Exception as e:
    raise RuntimeError(f"Failed to load features at {FEATURES_PATH}: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

def validate_and_align(records: List[dict]) -> pd.DataFrame:
    # Check keys exactly match saved features
    aligned = []
    for i, rec in enumerate(records):
        keys = list(rec.keys())
        missing = [c for c in features if c not in rec]
        extra = [k for k in keys if k not in features]
        if missing or extra:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "feature_mismatch",
                    "missing": missing,
                    "extra": extra,
                    "hint": "Provide exactly the features saved during training in reports/features.json, with numeric values."
                },
            )
        # Order columns per training feature order
        row = [float(rec[c]) for c in features]
        aligned.append(row)
    X = pd.DataFrame(aligned, columns=features)
    # Ensure numeric dtypes
    if not np.all([np.issubdtype(dt, np.number) for dt in X.dtypes.values]):
        raise HTTPException(status_code=400, detail="All feature values must be numeric (float).")
    return X

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Extract records dicts from Pydantic v2 RootModel schema (use `.root`)
    records = [item.root for item in req.records]
    if len(records) == 0:
        raise HTTPException(status_code=400, detail="Empty records list")
    X = validate_and_align(records)
    try:
        preds = model.predict(X).astype(int).tolist()
        probs = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            proba1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else None
            probs = proba1.tolist() if proba1 is not None else [float("nan")] * len(preds)
        else:
            probs = [float("nan")] * len(preds)
        return PredictResponse(preds=preds, probs=probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_failed: {e}")

