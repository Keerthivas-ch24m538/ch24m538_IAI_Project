# Titanic MLOps Pipeline: Spark + DVC + MLflow + FastAPI

A fully reproducible pipeline for the Kaggle Titanic classification task, featuring Spark preprocessing, DVC data/pipeline versioning, MLflow experiment tracking and registry, scikit-learn training, and a production-style FastAPI inference service packaged via Docker. The design emphasizes correctness, repeatability, and clear promotion from training to serving.[1][2][3][4]

## Highlights
- End-to-end DVC pipeline: get_data → validate_raw → preprocess_spark → validate_features → train → promote_model → serve artifacts.[2]
- Strict data checks: schema validation against YAML spec and numeric-only features guard.[5][3][6]
- MLflow tracking + optional registry: metrics, plots, and artifacts logged; optional model registration and stage transitions.[4]
- Deterministic serving: API consumes models/current.joblib and a locked features list from reports/features.json to enforce input schema.[7][1]
```
.
├── api/
│   ├── app.py                 # FastAPI service (loads models/current.joblib, reports/features.json)
│   └── Dockerfile             # Container for serving (uvicorn api.app:app)
├── configs/
│   └── mlflow.yaml            # Optional MLflow server config (SQLite + ./mlruns)
├── data/                      # DVC-tracked data (created by pipeline; not committed)
│   └── raw/                   # Downloaded CSV (get_data stage)
├── expectations/
│   └── schemas/
│       └── titanic.yaml       # Raw schema contract used by Spark validator
├── features/                  # Spark-processed features (Parquet) [DVC out]
├── models/
│   ├── baseline_logreg_pipeline.joblib
│   ├── tuned_logreg_pipeline.joblib
│   └── current.joblib         # Promoted winner used by API
├── reports/
│   ├── features.json          # Canonical feature order used by serving
│   ├── metrics.json           # Val/test metrics persisted for DVC/CI
│   └── plots/                 # Confusion matrix, ROC, PR curves
├── scripts/
│   ├── get_data.py            # Download dataset using params.yaml
│   ├── preprocess_spark.py    # Spark ETL + schema/type checks
│   ├── validate_raw_titanic.py
│   ├── validate_features.py
│   └── test_api_request.py    # Posts a real row to /predict
├── src/
│   ├── train.py               # MLflow-tracked training + optional registry
│   └── __init__.py            # (optional; if packaging src)
├── tests/
│   └── test_api.py            # Unit tests against api.app
├── dvc.yaml                   # Full pipeline: get_data → ... → promote_model
├── params.yaml                # Parameters: data URL, output dirs
├── requirements.txt           # All Python deps (DVC, MLflow, Spark, FastAPI, etc.)
└── README.md
```
## Repository layout
- dvc.yaml: Pipeline definition and stage wiring.[2]
- params.yaml: Centralized URLs and output dirs for data/preprocess stages.[8]
- scripts/: Data acquisition, validation, preprocessing, validation, and API test utility.[9][3][10][6][11]
- src/train.py: MLflow-tracked training, tuning (optional), metrics/plots logging, artifact persistence.[4]
- src/app.py, src/schemas.py: FastAPI service and request/response models, enforcing exact feature alignment.[12][1]
- reports/: Metrics, plots, and features.json (authoritative feature order).[4]
- models/: Saved sklearn pipelines and promoted current.joblib for serving.[7][2]

## System requirements
- Python 3.10 and Java (for Spark).[13]
- pip install -r requirements.txt (includes DVC, MLflow, PySpark, scikit-learn, FastAPI, uvicorn).[14]
- Docker (optional, for containerized serving).[13]

## Quickstart: full local reproduction
1) Setup environment
- Create and activate a virtual environment, then install dependencies.[14][13]

2) Reproduce the pipeline
- Run dvc repro to execute all stages: data download, raw validation, Spark preprocessing, feature validation, training, and model promotion (creates models/current.joblib).[2]
- Artifacts produced:
  - features/*.parquet (processed Spark features).[3]
  - reports/features.json (model feature order).[4]
  - reports/metrics.json, reports/plots/*.png (evaluation outputs).[4]
  - models/{baseline,tuned}_logreg_pipeline.joblib (trained models).[2][4]
  - models/current.joblib (promoted “winner” model for serving).[7]

3) Inspect metrics and plots
- Open reports/metrics.json for val/test metrics (accuracy, precision, recall, f1, roc_auc).[4]
- Review confusion matrix, ROC, and PR curves in reports/plots/.[4]

## MLflow tracking options
Two supported modes (no cloud required):

- File store mode (default)
  - Training logs MLflow runs to file:./mlruns.[4]
  - To browse, launch a local UI: mlflow ui --backend-store-uri file:./mlruns.[4]

- Local MLflow server + SQLite (optional)
  - Start a local server: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000.[15][13]
  - Export MLFLOW_TRACKING_URI=http://127.0.0.1:5000 before running training or dvc repro to log to the server; then open the UI at 127.0.0.1:5000.[15][4]

Registry note: train.py optionally registers the run’s model and can transition it to Staging/Production based on metrics; serving in this repo is joblib-based by default to keep evaluation simple.[4]

## DVC stages (what runs in dvc repro)
- get_data: Downloads titanic.csv to data/raw using params in params.yaml.[8][9][2]
- validate_raw: Validates columns, nulls, value ranges, and removes duplicates if PassengerId present.[11][2]
- preprocess: Spark pipeline casts types, indexes categorical columns, drops unused strings, validates against YAML schema, and writes Parquet features.[5][3][2]
- validate_features: Ensures numeric-only features and basic sanity checks on ranges.[6][2]
- train: Concatenates Parquet, splits train/val/test, trains baseline Logistic Regression (and optionally tunes), logs metrics/plots to MLflow, writes reports/features.json and reports/metrics.json, and saves model artifacts.[2][4]
- promote_model: Copies the selected winner from models/{baseline|tuned}_logreg_pipeline.joblib to models/current.joblib using the selection in reports/metrics.json. [2][7]

```
get_data            : python scripts/get_data.py
  └─ outs           : data/raw/

validate_raw        : python scripts/validate_raw_titanic.py
  └─ deps           : data/raw/titanic.csv

preprocess          : python scripts/preprocess_spark.py
  ├─ deps           : data/raw/, expectations/schemas/titanic.yaml
  └─ outs           : features/

validate_features   : python scripts/validate_features.py
  └─ deps           : features/

train               : python src/train.py
  ├─ deps           : features/, src/train.py
  ├─ outs           : models/baseline_logreg_pipeline.joblib
  │                   models/tuned_logreg_pipeline.joblib
  ├─ metrics        : reports/metrics.json (no-cache)
  └─ plots          : reports/plots/       (no-cache)

promote_model       : python scripts/promote_model.py
  ├─ deps           : models/{baseline,tuned}_logreg_pipeline.joblib
  │                   reports/metrics.json
  └─ outs           : models/current.joblib
```
## Serving the model (FastAPI)
The API expects:
- models/current.joblib (sklearn Pipeline with scaler + classifier).[1][7]
- reports/features.json (ordered feature names used during training).[1][4]

Run locally (without Docker):
- Ensure models/current.joblib and reports/features.json exist (run dvc repro first).[7][2]
- Start the server: uvicorn src.app:app --host 0.0.0.0 --port 8000.[1]

Endpoints:
- GET /health → {"status":"ok"}.[1]
- POST /predict with body {"records":[{feature_name: float, ...}]} → {"preds":[int,...], "probs":[float,...]}. The API enforces exact feature keys and numeric values.[12][1]

Example prediction request using a real feature row:
- Use scripts/test_api_request.py; it reads reports/features.json, picks one processed row, and posts to the API. Configure API_URL if needed.[10]

## Dockerized serving
Build:
- docker build -t titanic-api -f api/Dockerfile . (ensure the Dockerfile runs uvicorn with src.app:app and has PYTHONPATH set so src is importable).[13][1]

Run with artifacts mounted (recommended for evaluation):
- docker run --rm -p 8000:8000 -v $PWD/models/current.joblib:/app/models/current.joblib -v $PWD/reports/features.json:/app/reports/features.json titanic-api.[7][1]

Test:
- curl "http://127.0.0.1:8000/health".[1]
- python scripts/test_api_request.py (set API_URL to container host if needed).[10]

Common startup issue:
- Import fails on missing artifacts. Fix by running dvc repro first and mounting or baking models/current.joblib and reports/features.json into the image.[7][1]

## Testing
- Unit tests for API: tests import the FastAPI app and validate /health, valid predict, and feature mismatch errors. Align imports to src.app in tests if code resides under src/.[16][1]
- Integration test: scripts/test_api_request.py hits a running server and prints latency and response.[10]

## Configuration and parameters
- Data and preprocess parameters live in params.yaml (dataset URL, output dirs).[8]
- Training knobs use environment variables in src/train.py (RANDOM_SEED, TEST_SIZE, VAL_SIZE, DO_TUNE, LR_*); set them before dvc repro for reproducible experiments.[4]
- MLflow tracking mode is controlled by MLFLOW_TRACKING_URI; default is file:./mlruns.[4]

## Design choices
- Feature contract: reports/features.json written during training is the single source of truth for API input shape and order; the service strictly validates and aligns payloads to prevent silent skew.[1][4]
- Promotion policy: promote_model.py uses the selection in reports/metrics.json to choose the winner, keeping serving decoupled from MLflow registry for simpler local evaluation.[7][4]
- Safety checks: Raw and feature validations fail fast on schema/value anomalies to protect downstream training/serving.[6][11]

## Troubleshooting
- API import error: “Failed to load model/features”: ensure models/current.joblib and reports/features.json exist; re-run dvc repro, then mount or copy them into the container.[1][7]
- Feature mismatch error on /predict: payload keys must exactly match reports/features.json; use scripts/test_api_request.py to generate a compliant request.[10][1]
- No Parquet features found: verify preprocess and validate_features stages completed; check features/ directory contents.[3][6][2]
- MLflow runs not visible: confirm MLFLOW_TRACKING_URI matches the intended mode (file vs local server) before training.[15][4]

## What evaluators need to do
- Option A: Offline/file mode
  - pip install -r requirements.txt, dvc repro, uvicorn src.app:app, run scripts/test_api_request.py, inspect reports/metrics.json and reports/plots. No cloud services required.[14][10][2][1][4]
- Option B: Local MLflow server
  - Start mlflow server (SQLite + ./mlruns), export MLFLOW_TRACKING_URI, then dvc repro; browse runs at 127.0.0.1:5000; serve via uvicorn or Docker as above.[15][13][4]

## Roadmap (optional extensions)
- Registry-driven serving: Load models via MLflow registry URI (models:/titanic-logreg/Production) instead of joblib path; document promotion and rollback.[4]
- Drift detection and auto-retrain: Add a DVC stage to compute PSI/KS vs training baseline, produce reports/drift.json, and conditionally trigger retraining/promotion.[17]
- CI smoke tests: Automate dvc repro on a sample subset and run API tests in a container for continuous validation.[16][2]

## Maintainer
Keerthivas M -  ch24m538@smail.iitm.ac.in.[13]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/168a6e1f-830a-4968-917a-d1f5660270e3/app.py)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/e32017a7-b2cb-4303-9f6a-8399e8694f0c/dvc.yaml)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/7fd0b805-db94-4167-9426-62a99c221681/preprocess_spark.py)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/bb3a1800-8347-4d6f-8720-eba3ac07ccd5/train.py)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/2d8786dd-dda5-4109-b84f-a8a03d2305bb/titanic.yaml)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/76f227b8-596a-48f7-a397-f7d74007eaad/validate_features.py)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/484ac640-e73b-4510-9f54-1aaa506bcd2d/promote_model.py)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/2cf04dbc-67c0-47af-b696-78a8e376d8a6/params.yaml)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/c0c9993b-180e-4260-92c7-034fddb6748e/get_data.py)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/93c342ee-affa-43bb-a3a5-95fc7bbb6981/test_api_request.py)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/b800b840-c990-40c8-870b-72d575658f70/validate_raw_titanic.py)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/f3f61a7d-0e41-4c5b-b827-68730b9c1fc3/schemas.py)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/ad5f1c20-2c4a-4be3-86d3-8d8009fc8ec9/README.md)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/5c8a865e-d939-443a-bb3e-505d93996d29/requirements.txt)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/d30ff0b5-07e3-47eb-b67a-61541b66da4f/mlflow.yaml)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50464726/a3fadc23-4c98-483d-860b-59450e7f8d05/test_api.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_8a0e0074-6bf9-41c3-9fe8-8eef7cf38c65/51e6ff0f-79fc-40d4-991f-b8fbac87aa52/End_Term_Project.pdf)
