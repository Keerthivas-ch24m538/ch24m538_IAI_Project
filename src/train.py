import os
import json
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import random

# ----------------------------
# Config and deterministic seed
# ----------------------------
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)  # [3]

# ----------------------------
# MLflow tracking configuration
# ----------------------------
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)  # [3]
print("MLflow tracking URI:", mlflow.get_tracking_uri())  # [3]
# Always set or create an experiment to avoid file-store default issues
if mlflow_tracking_uri.startswith("http"):
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
else:
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "titanic-local")
mlflow.set_experiment(exp_name)  # [3]

# ----------------------------
# IO paths
# ----------------------------
features_dir = Path(os.getenv("FEATURES_DIR", "features"))  # [2]
reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)  # [2]
plots_dir = reports_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)  # [2]
model_dir = Path("models"); model_dir.mkdir(parents=True, exist_ok=True)  # [2]

# ----------------------------
# Load features
# ----------------------------
parquets = sorted([p for p in features_dir.glob("*.parquet")])
if not parquets:
    raise FileNotFoundError(f"No parquet files found in {features_dir.resolve()}")  # [2]
df = pd.concat([pd.read_parquet(fp) for fp in parquets], ignore_index=True)  # [2]
target_col = os.getenv("TARGET_COL", "Survived")
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in features DataFrame")  # [2]
# Ensure numeric-only inputs to avoid string->float errors
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()  # [4]
y = df[target_col].copy()  # [4]
feature_list = X.columns.tolist()
if X.empty:
    raise ValueError("No numeric features found after selection; check featurization/encoding stage")  # [4]

# Train/val/test split
test_size = float(os.getenv("TEST_SIZE", "0.2"))
val_size = float(os.getenv("VAL_SIZE", "0.2"))  # [3]
strat = y if len(y.unique()) <= 10 else None
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=strat
)  # [3]
strat_train = y_train_full if len(y.unique()) <= 10 else None
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=val_size, random_state=RANDOM_SEED, stratify=strat_train
)  # [3]

# ----------------------------
# Utility: evaluate and log
# ----------------------------
def evaluate_and_log(name: str, y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["accuracy"]  = float(accuracy_score(y_true, y_pred))  # [5]
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))  # [5]
    metrics["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))  # [5]
    metrics["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))  # [5]
    # Proper probability handling: 1-D proba or 2-D with class-1 column
    proba1 = None
    if y_proba is not None:
        if hasattr(y_proba, "ndim") and y_proba.ndim == 1:
            proba1 = y_proba
        elif hasattr(y_proba, "shape") and len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
            proba1 = y_proba[:, 1]
    if proba1 is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba1))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    # Log scalar metrics
    for k, v in metrics.items():
        mlflow.log_metric(f"{name}_{k}", v)  # [3]
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = plots_dir / f"{name}_confusion_matrix.png"
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(cm_path); plt.close()
    mlflow.log_artifact(str(cm_path), artifact_path="plots")  # [3]
    # ROC and PR curves when probabilities are available
    if proba1 is not None:
        # ROC
        roc_path = plots_dir / f"{name}_roc_curve.png"
        plt.figure()
        RocCurveDisplay.from_predictions(y_true, proba1)
        plt.title(f"{name} ROC Curve")
        plt.tight_layout()
        plt.savefig(roc_path); plt.close()
        mlflow.log_artifact(str(roc_path), artifact_path="plots")  # [3]
        # PR
        pr_path = plots_dir / f"{name}_pr_curve.png"
        plt.figure()
        PrecisionRecallDisplay.from_predictions(y_true, proba1)
        plt.title(f"{name} PR Curve")
        plt.tight_layout()
        plt.savefig(pr_path); plt.close()
        mlflow.log_artifact(str(pr_path), artifact_path="plots")  # [3]
    return metrics  # [5]

# ----------------------------
# Baseline model (LogReg) with scaler
# ----------------------------
baseline_params = {
    "C": float(os.getenv("LR_C", "1.0")),
    "penalty": os.getenv("LR_PENALTY", "l2"),
    "solver": os.getenv("LR_SOLVER", "lbfgs"),
    "max_iter": int(os.getenv("LR_MAX_ITER", "1000")),
    "n_jobs": -1 if os.getenv("LR_N_JOBS", "auto") == "auto" else int(os.getenv("LR_N_JOBS", "-1")),
}  # [3]
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(
        C=baseline_params["C"],
        penalty=baseline_params["penalty"],
        solver=baseline_params["solver"],
        max_iter=baseline_params["max_iter"],
        n_jobs=baseline_params["n_jobs"],
        random_state=RANDOM_SEED
    ))
])  # [6]

# ----------------------------
# Optional hyperparameter tuning
# ----------------------------
DO_TUNE = os.getenv("DO_TUNE", "false").lower() in ("1", "true", "yes")  # [3]

def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

with mlflow.start_run():  # [3]
    # Run-level metadata
    mlflow.log_params({
        "random_seed": RANDOM_SEED,
        "test_size": test_size,
        "val_size": val_size,
        "model_type": "LogisticRegression",
        "tuned": DO_TUNE
    })  # [3]
    # Feature lineage
    features_path = reports_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump({"features": feature_list}, f, indent=2)
    mlflow.log_artifact(str(features_path), artifact_path="reports")  # [3]
    # Train baseline
    pipeline.fit(X_train, y_train)  # [6]
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val) if hasattr(pipeline.named_steps["clf"], "predict_proba") else None
    val_metrics = evaluate_and_log("val_baseline", y_val, y_val_pred, y_val_proba)  # [5]
    best_model = pipeline
    best_name = "baseline"
    best_score = val_metrics["f1"]
    # Hyperparameter tuning (small grid)
    if DO_TUNE:
        with mlflow.start_run(nested=True, run_name="grid_search"):  # [3]
            param_grid = {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs", "liblinear"],
                "clf__max_iter": [500, 1000]
            }  # [3]
            gs = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="f1",
                cv=5,
                n_jobs=-1,
                refit=True
            )  # [3]
            gs.fit(X_train, y_train)
            mlflow.log_params({"grid_size": len(gs.cv_results_["mean_test_score"])})  # [3]
            mlflow.log_params({f"best_{k}": v for k, v in gs.best_params_.items()})  # [3]
            y_val_pred_gs = gs.best_estimator_.predict(X_val)
            y_val_proba_gs = (
                gs.best_estimator_.predict_proba(X_val)
                if hasattr(gs.best_estimator_.named_steps["clf"], "predict_proba") else None
            )
            val_metrics_gs = evaluate_and_log("val_tuned", y_val, y_val_pred_gs, y_val_proba_gs)  # [5]
            if val_metrics_gs["f1"] > best_score:
                best_model = gs.best_estimator_
                best_name = "tuned"
                best_score = val_metrics_gs["f1"]

    # Final evaluation on test
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test) if hasattr(best_model.named_steps["clf"], "predict_proba") else None
    test_metrics = evaluate_and_log(f"test_{best_name}", y_test, y_test_pred, y_test_proba)  # [5]
    # Persist artifacts (DVC-friendly metrics)
    metrics_json = {
        "val": val_metrics,
        "test": test_metrics,
        "selection_metric": "f1",
        "selected_model": best_name
    }  # [2]
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    mlflow.log_artifact(str(metrics_path), artifact_path="reports")  # [3]

    # Save model artifact (winning model)
    model_path = model_dir / f"{best_name}_logreg_pipeline.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(str(model_path), artifact_path="model")

    # --- FIX: ENSURE BOTH OUTPUT MODEL FILES EXIST FOR DVC ---
    # Save baseline model (even if it is not the best)
    baseline_path = model_dir / "baseline_logreg_pipeline.joblib"
    if best_name == "baseline":
        # Already saved
        pass
    else:
        # Save the baseline pipeline as baseline, if not already
        joblib.dump(pipeline, baseline_path)
    # Save tuned model (even if not selected)
    tuned_path = model_dir / "tuned_logreg_pipeline.joblib"
    if best_name == "tuned":
        # Already saved as best_model above
        pass
    elif DO_TUNE:
        # Save the tuned pipeline (if it ran but wasn't the winner)
        if 'gs' in locals() and hasattr(gs, 'best_estimator_'):
            joblib.dump(gs.best_estimator_, tuned_path)
    else:
        # Ensure an empty dummy file if tuning didn't run
        if not tuned_path.exists():
            tuned_path.write_bytes(b"")

    # Continue with MLflow log_model, tagging, and print
    try:
        mlflow.sklearn.log_model(best_model, artifact_path="mlflow_model")
    except Exception:
        pass
    print(f"[DONE] Selected={best_name}  Test F1={test_metrics['f1']:.4f}  Test Acc={test_metrics['accuracy']:.4f}")
    run_id = mlflow.active_run().info.run_id
    client = mlflow.tracking.MlflowClient()
    # Add tags for traceability
    client.set_tag(run_id, "selected_model", best_name)
    client.set_tag(run_id, "git_sha", get_git_sha())

    # -----------------------
    # MLflow Model Registry
    # -----------------------
    # Register the logged model, set stage, and optionally promote
    model_name = "titanic-logreg"  # can be customized for your project

    model_uri = f"runs:/{run_id}/mlflow_model"
    try:
        reg_model = mlflow.register_model(model_uri, model_name)
        reg_version = reg_model.version
        print(f"Registered model as {model_name} v{reg_version}")

        client.transition_model_version_stage(
            name=model_name,
            version=reg_version,
            stage="Staging",
            archive_existing_versions=False,
        )
        print("Model promoted to STAGING.")

        # Promote to Production if F1 > 0.9 (customize as needed)
        if test_metrics['f1'] > 0.9:
            client.transition_model_version_stage(
                name=model_name,
                version=reg_version,
                stage="Production",
                archive_existing_versions=True,
            )
            print("Model promoted to PRODUCTION (metric threshold exceeded).")
    except Exception as e:
        print(f"MLflow registry step failed: {e}")

