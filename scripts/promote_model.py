import json
from pathlib import Path
import shutil

# Load which model was selected as best
with open("reports/metrics.json") as f:
    metrics = json.load(f)

winner = metrics.get("selected_model", "baseline")  # default to baseline if missing
src = f"models/{winner}_logreg_pipeline.joblib"
dst = "models/current.joblib"
shutil.copyfile(src, dst)
print(f"Promoted {src} -> {dst}")

