```markdown
# 🏆 Titanic ML MLOps Pipeline

This repository demonstrates a reproducible ML pipeline for the Titanic dataset using **DVC** for data and pipeline versioning, **MLflow** for experiment tracking, **PySpark** for preprocessing, and **FastAPI + Docker** for lightweight, scalable API deployment.

---

## 📁 Project Structure

```
.
├── api/                  # FastAPI app & Dockerfile
├── configs/              # Project configs (MLflow, etc.)
├── data/                 # DVC-tracked data (not in Git)
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # All pipeline parameters (urls, etc.)
├── requirements.txt      # Python dependencies
├── scripts/              # Data download & preprocessing scripts
├── features/             # Preprocessed features (DVC-managed)
└── README.md
```

---

## 🛠️ Environment Setup

### Option 1: Conda (Recommended)

```
conda create -n mlops-pipeline python=3.10 -y
conda activate mlops-pipeline
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Virtualenv

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📊 MLflow Tracking Setup

**Start a local MLflow tracking server (for experiment logging):**
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```
Open the UI at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🗄️ DVC Data Versioning (Remote, Optional)

**Set up a local DVC remote (if not done already):**
```
mkdir -p dvc_store
dvc remote add -d local ./dvc_store
```

---

## 🚦 Pipeline Reproducibility

Reproduce the full pipeline (download & preprocess):
```
dvc repro
```

**Push outputs to DVC remote:**
```
dvc push
```

---

## 🚢 Build & Launch the API (FastAPI + Docker)

### Build the Docker image:
```
docker build -t titanic-api -f api/Dockerfile .
```

### Run the container:
```
docker run -p 8000:8000 titanic-api
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to verify the API is running.

---

## 🧑‍💻 Pipeline Details

- **get_data:** Downloads the Titanic CSV to `data/raw` using parameters in `params.yaml`
- **preprocess (PySpark):** Reads from `data/raw/`, cleans and encodes features, writes processed data to `features/` in Parquet format

All **parameter values** (like URLs, output directories) are managed in `params.yaml`.

---

## 🔌 API Scaffold Example

**File:** `api/app.py`
```
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is up and running"}
```
_Real model inference endpoints can be added as the project grows._

---

## 🧹 Repo Hygiene & Submission Checklist

- [x] All pipeline/config files tracked: `requirements.txt`, `configs/mlflow.yaml`, `dvc.yaml`, `params.yaml`, `api/Dockerfile`, scripts
- [x] Large data, model outputs only tracked by DVC (never Git)
- [x] All code & commands documented for rapid demo
- [x] Minimal API endpoint available via Docker

---

## 💡 Demo Flow (10 minutes)

1. Clone repo & enter project folder
2. Set up environment (`conda ...` or `venv ...`)
3. (Optional) Start `mlflow server` in a new terminal
4. `dvc pull` (if collaborating)
5. `dvc repro` (runs pipeline)
6. `dvc push` (pushes data to remote)
7. `docker build` and `docker run` the API
8. Open browser to API root endpoint
9. (Optional) View MLflow UI for experiment logs

---

## 📝 Contact

Maintainer: **[Keerthivas M]**  
Email: [ch24m538@smail.iitm.ac.in]

---
```

