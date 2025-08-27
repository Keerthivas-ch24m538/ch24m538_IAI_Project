from typing import Dict, List
from pydantic import BaseModel, Field, RootModel

class PredictItem(RootModel):
    root: Dict[str, float]

class PredictRequest(BaseModel):
    records: List[PredictItem] = Field(..., description="Batch of feature mappings")

class PredictResponse(BaseModel):
    preds: List[int]
    probs: List[float]

