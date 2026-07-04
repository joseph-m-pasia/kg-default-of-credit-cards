import pandas as pd
import numpy as np
import pytest

from pkg_credit_default.modeling.trainer import train_model
from pkg_credit_default.utils.utils import load_ml_model


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "LIMIT_BAL": [20000, 120000, 50000, 80000],
        "AGE": [25, 45, 34, 50],
        "PAY_0": [0, 2, -1, 1],

        "BILL_AMT1": [500, 2000, 1500, 3000],
        "BILL_AMT2": [600, 2100, 1600, 3100],
        "BILL_AMT3": [700, 2200, 1700, 3200],
        "BILL_AMT4": [800, 2300, 1800, 3300],
        "BILL_AMT5": [900, 2400, 1900, 3400],
        "BILL_AMT6": [1000, 2500, 2000, 3500],

        "PAY_AMT1": [200, 500, 300, 400],
        "PAY_AMT2": [250, 550, 350, 450],
        "PAY_AMT3": [300, 600, 400, 500],
        "PAY_AMT4": [350, 650, 450, 550],
        "PAY_AMT5": [400, 700, 500, 600],
        "PAY_AMT6": [450, 750, 550, 650],
    })

    y = pd.Series([0, 1, 0, 1], name="target")
    return X, y


@pytest.fixture
def sample_config(tmp_path):
    """
    Minimal config required for training.
    """
    return {
        "models": {
            "logistic_regression": {
                "class": "sklearn.linear_model.LogisticRegression",
                "params": {"max_iter": 200},
                "param_grid": {}
            }
        },
        "selection": {"primary_metric": "accuracy"},
        "gridCV": {"cv": 2, "n_jobs": 1, "verbose": 0},
        "metrics": "accuracy",
        "paths": {
            "output_dir_models": str(tmp_path / "models")
        }
    }


def test_training_pipeline_end_to_end(sample_data, sample_config):
    """
    End-to-end test:
    - trains the pipeline
    - verifies feature_names_in_
    - saves the model
    - loads the model
    - performs prediction
    """
    X_train, y_train = sample_data

    # Train
    result = train_model(
        X_train=X_train,
        y_train=y_train,
        config=sample_config,
        model_type="logistic_regression",
        save_output=True
    )

    model = result["model"]

    #  Feature names must exist
    assert hasattr(model, "feature_names_in_"), "feature_names_in_ missing"
    assert len(model.feature_names_in_) > 0, "feature_names_in_ is empty"

    # Saved model must contain feature names
    loaded = load_ml_model(result["model_path"])
    assert loaded["feature_names"] is not None, "Saved feature_names is None"
    assert len(loaded["feature_names"]) > 0, "Saved feature_names is empty"

    # Prediction must work end-to-end
    sample = X_train.iloc[[0]]
    pred = loaded["model"].predict(sample)
    proba = loaded["model"].predict_proba(sample)

    assert pred.shape == (1,), "Prediction shape incorrect"
    assert proba.shape == (1, 2), "Probability shape incorrect"
