from pkg_credit_default.utils.logger import logger
from datetime import datetime

import os
import joblib


def save_model(model, model_dir, model_type="", timestamp=True) -> str:
    """
    Save the trained model and feature names to disk.
    """
    logger.info(f"Saving model '{model_type}' to disk...")

    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""

    model_dir = os.path.join(
        model_dir, tstamp + f"_{model_type}"
    )  # e.g. "outputs/models/20240101_120000_logistic_regression"
    os.makedirs(model_dir, exist_ok=True)  # create folder if it doesn't exist

    model_path = os.path.join(model_dir, f"{model_type}_model.pkl")

    # Extract feature names from the trained pipeline
    try:
        feature_names = list(model.feature_names_in_)
        logger.info(f"Extracted {len(feature_names)} feature names from model.")
    except AttributeError:
        feature_names = None
        logger.warning("Model has no attribute 'feature_names_in_'. Feature names will not be saved.")

    # Save both model and feature names together
    bundle = {
        "model": model,
        "feature_names": feature_names
    }

    joblib.dump(bundle, model_path)

    logger.info(f"Model saved to {model_path}")

    return model_path


def load_ml_model(model_path: str = None):
    """
    Load a trained ML model from disk.
    Asumes the model is saved as a .pkl file using joblib.
    Args:    model_path (str): The path to the saved model file.
    Returns: The loaded model object, or None if the file does not exist.
    """
    logger.info("Loading model and its metrics...")

    if os.path.exists(model_path):
        ml_model = joblib.load(model_path)
        return ml_model
    else:
        logger.warning(f"Model not found at {model_path}. Returning None.")
        return None
