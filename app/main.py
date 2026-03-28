from fastapi import FastAPI, HTTPException
from joblib import dump, load
from pydantic import BaseModel, field_validator
import math
from typing import List
import logging, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = load('app/model.joblib')

app = FastAPI()

EXPECTED_FEATURES = 7
class PredictionInput(BaseModel):
    input_array : List[List[float]]

    @field_validator("input_array")
    def validate_input(cls, v):
        if not v:
            raise ValueError("input_array cannot be empty")

        for row in v:
            if len(row) != EXPECTED_FEATURES:
                raise ValueError(f"Each row must have {EXPECTED_FEATURES} features")

            for val in row:
                if math.isnan(val):
                    raise ValueError("NaN not allowed")

        return v

@app.get("/health")
def get_health():
    return {"status":"ok"}

@app.post("/predict")
def predict(data: PredictionInput):    
    try:
        logger.info(f"Input array length {len(data.input_array)}")
        res = model.predict(data.input_array) #([[2001, 2, 4, 6, 7, 0, 2], [2001, 2, 4, 6, 7, 1, 2]])
        return {"prediction":res.tolist()}
    except Exception as e:
        logger.exception("Prediction Failed")
        raise HTTPException(status_code=500, detail="Inference Failed")


# Production Ready
# 1) Input Validation -
# 2) Health Checks - Status OK
# 3) Logging - P2
# 4) Unit Test / API Test - P1
# 5) Security - P3