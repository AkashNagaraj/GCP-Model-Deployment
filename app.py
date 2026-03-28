from fastapi import FastAPI 
from joblib import dump, load
from pydantic import BaseModel
from typing import List

model = load('model.joblib')

app = FastAPI()

class PredictionInput(BaseModel):
    input_array : List[List[float]]

@app.get("/")
def test():
    return "Hello World"

@app.post("/predict")
def predict(data: PredictionInput):    
    res = model.predict(data.input_array) #([[2001, 2, 4, 6, 7, 0, 2], [2001, 2, 4, 6, 7, 1, 2]])
    return {"res":res.tolist()}

