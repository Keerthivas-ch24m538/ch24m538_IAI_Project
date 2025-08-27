import pandas as pd
import numpy as np
import glob
import sys

parquet_files = glob.glob("features/*.parquet")
df = pd.concat([pd.read_parquet(fp) for fp in parquet_files], ignore_index=True)

# Feature should have only numeric columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    sys.exit(f"ERROR: Non-numeric columns in features: {non_numeric_cols}")

# Value checks
if (df["Fare"] < 0).any():
    sys.exit("ERROR: Some Fare values are negative!")
if (df["Fare"] > df["Fare"].quantile(0.99)).any():
    print("Warning: Some Fare values exceed 99th percentile cap.")
# Optionally cap/winsorize Fare for downstream safety:
# df["Fare"] = df["Fare"].clip(lower=0, upper=df["Fare"].quantile(0.99))

if (df["Age"] < 0).any() or (df["Age"] > 100).any():
    print("Warning: Age values out of 0-100 range detected.")

print("Feature data validation (numeric guard) passed.")

