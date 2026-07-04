from sklearn.compose import ColumnTransformer

from pkg_credit_default.utils.logger import logger
from pkg_credit_default.utils.utils import save_model
from pkg_credit_default.features.feature_builder import FeatureEngineering

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector 

import importlib
import pandas as pd
import numpy as np
from joblib import Memory
from typing import Any, Dict


def get_model_class_and_params(config: Dict, model_type: str, save_model: bool = True):
    """
    Load the model class dyna mically from config.
    """
    logger.info(f"Loading model class and parameters for '{model_type}' from config...")

    model_config = config["models"][model_type]  # Get the specific model block

    class_path = model_config["class"]  # e.g. "sklearn.linear_model.LogisticRegression"

    # Dynamically import the class
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    # Get default parameters
    params = model_config.get("params", {})

    return ModelClass, params, model_config

def build_pipeline(model):
    """
    Build a pipeline with feature engineering, preprocessing, and the model.
    """
    logger.info("Building pipeline...")

    # Define the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),  
            make_column_selector(dtype_include=np.number),  )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    memory = Memory(location="cache/sklearn", verbose=10)

    # Create the full pipeline
    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineering(n_months=6)),  # Add feature engineering step
            ("preprocessor", preprocessor),                           # Apply preprocessing
            ("model", model),                                         # Add the model
        ],
        # Cache intermediate results to speed up GridSearchCV
        memory=memory   
    )

    return pipeline


def train_model(X_train, 
                y_train, 
                config, 
                model_type="logistic_regression", 
                save_output=True) -> Dict[str, Any]:

    logger.info(f"Training {model_type} model...")

    model_type = model_type.lower()

    # ======================= Load Model Dynamically ================

    ModelClass, default_params, model_config = get_model_class_and_params(config, model_type)

    # Create model instance with default params
    regressor = ModelClass(**default_params)

    param_grid = model_config.get("param_grid", {})
    
    pipeline = build_pipeline(regressor)

    logger.info("Pipeline steps:")
    for name, step in pipeline.steps:
        logger.info(f"  {name:20}: {step.__class__.__name__}")

    # ======================= Grid Search =======================
    logger.info("Performing GridSearchCV...")

    metric = config["selection"]["primary_metric"]
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=config["gridCV"]["cv"],
        n_jobs=config["gridCV"]["n_jobs"],
        verbose=config["gridCV"]["verbose"],
        scoring=config["metrics"],
        refit=metric,
    )

    grid_search.fit(X_train, y_train)
    logger.info(f"Model training of {model_type} completed.")

    best_model = grid_search.best_estimator_
   
    # ======================= Logging Results =======================

    best_idx = grid_search.best_index_
    std_score = grid_search.cv_results_[f"std_test_{metric}"][best_idx]

    logger.info(f"Best CV Score ({metric})    : {grid_search.best_score_:.4f}")
    logger.info(f"CV Score Std Dev ({metric}) : {std_score:.4f}")
    logger.info(f"Training Score ({metric})   : {best_model.score(X_train, y_train):.4f}")
    logger.info(f"Best Parameters             : {grid_search.best_params_}")

    # ======================= Save Model ============================
    if save_output:
        model_dir = config["paths"]["output_dir_models"]
        model_path = save_model(best_model, model_dir, model_type)

    return {
        "model": best_model,
        "best_score": grid_search.best_score_,
        "best_params": grid_search.best_params_,
        "model_dir": model_dir if save_output else None,
        "model_path": model_path if save_output else None,
        "grid_search": grid_search,
    }


# =========================
# EXAMPLE TRAINING
# =========================
if __name__ == "__main__":

    df = pd.DataFrame({
        "BILL_AMT1": [100, 200, 150, 250, 300],
        "PAY_AMT1": [20, 50, 30, 60, 70],
        "LIMIT_BAL": [1000, 2000, 1500, 2500, 3000],
        "PAY_1": [1, 0, 1, 0, 1],
        "target": [0, 1, 0, 1, 0],
    })

    X = df.drop(columns=["target"])
    y = df["target"]
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = build_pipeline(model)

    grid = GridSearchCV(
        pipeline,
        param_grid={},
        cv=2,
        scoring="accuracy",
        refit=True
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_

    # =========================
    # FEATURE NAMES (FINAL)
    # =========================
    X_fe = best_model.named_steps["feature_engineering"].transform(X)

    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out(
        X_fe.columns
    )

    print("\nFINAL FEATURES:")
    print(feature_names)