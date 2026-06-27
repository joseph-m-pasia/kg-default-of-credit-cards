from fastapi import FastAPI
import joblib
import pandas as pd

from app.schemas import (
    PredictionRequest,
    PredictionResponse
)

app = FastAPI()

model = None


def get_model():
    global model

    if model is None:
        model = joblib.load("artifacts/model.pkl")

    return model


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    data: PredictionRequest
):

    model = get_model()

    df = pd.DataFrame(
        [data.model_dump()]
    )

    prediction = model.predict(df)[0]

    probability = (
        model
        .predict_proba(df)[0][1]
    )

    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability)
    )   