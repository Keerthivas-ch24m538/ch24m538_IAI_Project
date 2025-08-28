# scripts/test_api_request.py
import os
import json
from pathlib import Path
import time
import pandas as pd
import numpy as np
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # change if container mapped differently

ROOT = Path(__file__).resolve().parents[1]
FEATURES_JSON = ROOT / "reports" / "features.json"
FEATURES_DIR = ROOT / "features"

def load_feature_names():
    if not FEATURES_JSON.exists():
        raise FileNotFoundError(f"Missing {FEATURES_JSON}. Run training to generate it.")
    with open(FEATURES_JSON, "r") as f:
        data = json.load(f)
    feats = data.get("features", [])
    if not feats:
        raise ValueError("features.json does not contain a non-empty 'features' list.")
    return feats

def pick_one_row_df():
    if not FEATURES_DIR.exists():
        raise FileNotFoundError(f"Missing {FEATURES_DIR}. Ensure preprocess stage has produced parquet features.")
    # pick most recent parquet file
    parquets = sorted(FEATURES_DIR.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    if not parquets:
        raise FileNotFoundError(f"No parquet files in {FEATURES_DIR}.")
    fp = parquets[-1]
    df = pd.read_parquet(fp)
    if df.empty:
        raise ValueError(f"Selected parquet {fp} is empty.")
    return df.iloc[[0]].copy()  # keep as DataFrame with one row

def to_payload(df_row: pd.DataFrame, feature_names):
    # ensure only model features and correct order
    present = [c for c in feature_names if c in df_row.columns]
    missing = [c for c in feature_names if c not in df_row.columns]
    if missing:
        # Try selecting numeric columns and fallback to zeros for missing (last resort)
        raise ValueError(f"Row missing required features: {missing}. Check preprocessing alignment.")
    row = df_row[feature_names].iloc[0]
    # Convert to floats (API expects numbers)
    rec = {k: (float(v) if pd.notna(v) else 0.0) for k, v in row.to_dict().items()}
    return {"records": [rec]}

def main():
    feature_names = load_feature_names()
    df_row = pick_one_row_df()
    payload = to_payload(df_row, feature_names)

    url = f"{API_URL}/predict"
    print(f"POST {url}")
    print("Payload:", json.dumps(payload, indent=2)[:600], "...\n")

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=30)
    dt = (time.time() - t0) * 1000.0

    print("Status:", resp.status_code)
    try:
        print("Response JSON:", json.dumps(resp.json(), indent=2))
    except Exception:
        print("Raw response:", resp.text)
    print(f"Latency: {dt:.1f} ms")

if __name__ == "__main__":
    main()

