from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Titanic API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

# Fields aligned with a typical preprocessed Titanic feature set
class PredictRequest(BaseModel):
    Pclass: int
    Sex: float        # encoded: e.g., female=0.0, male=1.0 (match your preprocessing!)
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: float   # encoded: e.g., S=2.0, C=0.0, Q=1.0 (match your preprocessing!)

@app.post("/predict")
def predict(req: PredictRequest):
    # For now, return a stub using a simple rule or placeholder
    # Replace this with: load model + np.array([[...features...]]) -> model.predict(...)
    features = np.array([[req.Pclass, req.Sex, req.Age, req.SibSp, req.Parch, req.Fare, req.Embarked]])
    return {
        "prediction": 0,
        "received_features": features.tolist(),
        "note": "Stub prediction. Wire up trained model next."
    }

