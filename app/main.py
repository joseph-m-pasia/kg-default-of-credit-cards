from fastapi import FastAPI
import joblib
import pandas as pd

from app.schemas import (
    PredictionRequest,
    PredictionResponse
)

app = FastAPI()

model_bundle = None

def get_model():

    global model_bundle
    if model_bundle is None:
        # Load the bundle saved by save_model()
        model_bundle = joblib.load("artifacts/model.pkl")

        print("Model loaded successfully.")
        print("Bundle keys:", model_bundle.keys())  # should show: model, feature_names

    # Return both the model and feature names
    return model_bundle["model"], model_bundle["feature_names"]

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    data: PredictionRequest
):
    # load the model and feature names
    model, feature_names = get_model()

    # Create a DataFrame from the input data
    df = pd.DataFrame(
        [data.model_dump()]
    )

    # Ensure the DataFrame has the same columns as the model was trained on
    df = df[feature_names]

    # Predict the outcome
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability)
    )   