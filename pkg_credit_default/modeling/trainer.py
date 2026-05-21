from datetime import datetime
import os
from pyexpat import model 
import joblib

from pkg_credit_default.utils.logger import logger
from pkg_credit_default.features.feature_builder import FeatureEngineering

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

import importlib
from typing import Any, Dict

def get_model_class_and_params(config: Dict, model_type: str, save_model: bool = True):
    """
    Load the model class dyna mically from config.
    """
    logger.info(f"Loading model class and parameters for '{model_type}' from config...")
    
    model_config = config["models"][model_type]             # Get the specific model block

    class_path = model_config["class"]                      # e.g. "sklearn.linear_model.LogisticRegression"
    
    # Dynamically import the class
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)
    
    # Get default parameters
    params = model_config.get("params", {})

    return ModelClass, params, model_config

def save_model(model, model_dir, model_type="", timestamp=True):
    """
    Save the trained model to disk.
    """
    logger.info(f"Saving model '{model_type}' to disk...")

    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""

    model_dir = os.path.join(model_dir, tstamp + f"_{model_type}")  # e.g. "outputs/models/20240101_120000_logistic_regression"
    os.makedirs(model_dir, exist_ok=True)                           # create folder if it doesn't exist

    model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
    joblib.dump(model, model_path)
  
    logger.info(f"Model saved to {model_path}")

    return model_path    

def train_model(X_train, y_train, config, model_type="logistic_regression", save_output = True):

    logger.info(f"Training {model_type} model...")

    model_type = model_type.lower()

    # ======================= Load Model Dynamically ================

    ModelClass, default_params, model_config = get_model_class_and_params(config, model_type)

    # Create model instance with default params
    regressor = ModelClass(**default_params)
    
    param_grid = model_config.get("param_grid", {})

    # ======================= Create Pipeline =======================
    pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineering(n_months=6)),  # Add feature engineering step
        ("imputer", SimpleImputer(strategy="median")),            # Handle missing values
        ("scaler", StandardScaler()),                             # Scale features
        ("model", regressor)                                      # Add the model
    ])

    logger.info("Pipeline steps:")
    for name, step in pipeline.steps:
        logger.info(f"  {name:20}: {step.__class__.__name__}")

    # ======================= Grid Search =======================
    logger.info("Performing GridSearchCV...")
    
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grid, 
                               cv=config["gridCV"]["cv"], 
                               n_jobs=config["gridCV"]["n_jobs"], 
                               verbose=config["gridCV"]["verbose"]
                               )

    grid_search.fit(X_train, y_train)
    logger.info("Model training completed.")

    best_model = grid_search.best_estimator_
    
    # ======================= Logging Results =======================
    logger.info(f"Best CV Score    : {grid_search.best_score_:.4f}")
    logger.info(f"CV Score Std Dev : {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f}")
    logger.info(f"Training Score   : {best_model.score(X_train, y_train):.4f}")
    logger.info(f"Best Parameters  : {grid_search.best_params_}")
      
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
        "grid_search": grid_search
    }

############### RUN TRAINING PIPELINE ###############
if __name__ == "__main__":

    # STEP 0: SELECT MODEL TYPE
    model_type = "xgb_regressor"  # Change this to select different model (e.g. "logistic_regression", "random_forest", "xgb_regressor")

    # STEP 1: CONFIG
    config = load_config()

    # STEP 2: RAW DATA
    df = load_data_from_csv(config["data"]["raw_path"])

    # STEP 3: DATA CLEANING
    df = clean_data(df)

    # STEP 5: REMOVE VARIABLES
    df = remove_variables(df, config)

    print(df.columns)

    X_train = df.drop(config["data"]["target_variable"], axis=1)
    y_train = df[config["data"]["target_variable"]]

    model_results = train_model(X_train, y_train, config, model_type=model_type)

