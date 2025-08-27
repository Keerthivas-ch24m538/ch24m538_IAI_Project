```markdown
# Titanic ML MLOps Pipeline

This repository demonstrates a reproducible, production-style ML pipeline with DVC for data/version control, MLflow for experiment tracking, PySpark for preprocessing, and FastAPI+Docker for API deployment.

---

## 📁 Project Structure

```
.
├── api/                  # API code and Dockerfile
├── configs/              # Config files (e.g., mlflow.yaml)
├── data/                 # DVC-tracked data (not in Git)
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Pipeline parameters (urls, file paths, etc.)
├── requirements.txt      # Python dependencies
├── scripts/              # ML/data scripts
├── features/             # Preprocessed feature outputs (via DVC)
└── README.md
```

---

## 🔥 Quickstart: Run the Whole Pipeline

### 1. **Clone and enter repository**
```
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. **Environment Setup**

**Using Conda:**
```
conda create -n mlops-pipeline python=3.10 -y
conda activate mlops-pipeline
pip install --upgrade pip
pip install -r requirements.txt
```

**Or Using venv/pip:**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. **MLflow Tracking (Optional, but recommended for full pipeline)**

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```
- Visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 4. **DVC Data Versioning**

If using a local DVC remote:
```
mkdir -p dvc_store
dvc remote add -d local ./dvc_store
```

---

### 5. **Pipeline Reproduction**

```
dvc pull            # If collaborating & remote is set
dvc repro           # Runs data download and preprocessing with PySpark
dvc push            # Pushes latest outputs to DVC remote
```

---

### 6. **Build and Launch the API (Docker + FastAPI)**

```
docker build -t titanic-api -f api/Dockerfile .
docker run -p 8000:8000 titanic-api
```
Open your browser to: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 🔧 Pipeline Details

**Stages:**

1. **get_data:**  
   Downloads Titanic CSV to `data/raw` using parameters in `params.yaml`.

2. **preprocess (PySpark):**  
   Reads `data/raw`, cleans and encodes features, writes `features/` in Parquet format.
   - See `scripts/preprocess_spark.py`.

**Note:**  
- Output, input paths, and URLs are in `params.yaml` at the repo root.

---

## 📦 API Scaffold

**File:** `api/app.py`
```
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is up and running"}
```
REST endpoints will be added to serve predictions in later project phases (e.g., loading models from MLflow registry).

---

## 💾 Committed, Tracked and Ignored Files

Committed:
- `requirements.txt`, `configs/mlflow.yaml`, `dvc.yaml`, `params.yaml`, `api/Dockerfile`, scripts

Ignored (see `.gitignore`, `.dvcignore`):
- Data/artifact directories: `data/`, `features/`, `dvc_store/`, large binaries, etc.

---

## ✅ Demo Checklist

- [ ] Clone and set up Python env
- [ ] (Optional) Run MLflow server
- [ ] Run: `dvc repro`
- [ ] (Optional) Run: `dvc push`
- [ ] Run: `docker build ...` and `docker run ...`
- [ ] Open browser to API root endpoint

---

## 📝 Notes

- All data versioned via DVC; only `*.dvc` pointers and pipeline defs in Git.
- MLflow used for experiment tracking and (later) model registry.
- PySpark-based preprocessing is reproducible, modular.
- API is ready to serve/test, easy to extend for model inference.

---

_Contact: [Keerthivas M] ([ch24m538@smail.iitm.ac.in])_
```


