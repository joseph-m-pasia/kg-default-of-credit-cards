'''
Description: Main FastAPI application for credit default prediction.
This application provides endpoints for health checks and making predictions 
using a trained machine learning model.

To execute locally: 
1. Run the following command in the terminal: uvicorn app.main:app --reload
2. Open the browser and navigate to http://127.0.0.1:8000/docs # to view the API documentation and test the endpoints.
3. Use the /predict endpoint to send a POST request with the required input data for prediction.
4. The response will include the prediction, probability, and risk category.
'''



from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from pathlib import Path

from pkg_credit_default.utils.logger import logger
from pkg_credit_default.utils.utils  import load_ml_model

from app.schemas import (
    PredictionRequest,
    PredictionResponse
)

from app.schemas.defaults import DEFAULT_FEATURES

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_PATH = Path("artifacts/model.pkl")

model_bundle = None


# -----------------------------------------------------------------------------
# FastAPI lifespan
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model when the API starts.
    """
    get_model()
    yield


app = FastAPI(
    title="Credit Default Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

model_bundle = None


# -----------------------------------------------------------------------------
# Health endpoint
# -----------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "i am healthy"
    }

# -----------------------------------------------------------------------------
# Root endpoint
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Credit Default Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def get_model():

    logger.info("Loading model and feature names...")   
    global model_bundle
    if model_bundle is None:
        # Load the bundle saved by save_model()
        model_bundle = load_ml_model(MODEL_PATH)

    # Return both the model and feature names
    return model_bundle["model"][1]['model'], model_bundle["feature_names"]


# -----------------------------------------------------------------------------
# Prediction endpoint
# -----------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)


def predict(data: PredictionRequest):
    """
    Endpoint to make predictions using the trained model.
    Args:
        data (PredictionRequest): Input data for prediction.
    Returns:
        PredictionResponse: The prediction result and probability.
     """

    logger.info("Received prediction request...")

    # load the model and feature names
    model, feature_names = get_model()

    try: 

        # Create a DataFrame from the input data
        input_data = data.model_dump()
        input_data.update(DEFAULT_FEATURES)  # Merge with default features

        df = pd.DataFrame(
            [input_data]
        )
        
        # Ensure the DataFrame has the same columns as the model was trained on
        if feature_names is not None:
            df = df[feature_names]

        # Predict the outcome
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        # Simple risk categorization based on probability
        if probability >= 0.5:
            risk_category = "High Risk"
        else:
            risk_category = "Low Risk"

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_category=risk_category
        )   

    except Exception as e:
        logger.error(f"Error occurred while making prediction: {e}")
        raise HTTPException(status_code=500, detail="Error occurred while making prediction")

