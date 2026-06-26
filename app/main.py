# app/main.py

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("artifacts/model.pkl")

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }