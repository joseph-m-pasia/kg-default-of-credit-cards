from pkg_credit_default.utils.logger import logger
from pkg_credit_default.utils.utils import save_model
from pkg_credit_default.features.feature_builder import FeatureEngineering

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

import importlib
from joblib import Memory
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

def train_model(X_train, y_train, config, model_type="logistic_regression", save_output = True) -> Dict[str, Any]:

    logger.info(f"Training {model_type} model...")

    model_type = model_type.lower()

    # ======================= Load Model Dynamically ================

    ModelClass, default_params, model_config = get_model_class_and_params(config, model_type)

    # Create model instance with default params
    regressor = ModelClass(**default_params)
    
    param_grid = model_config.get("param_grid", {})

    # ======================= Create Pipeline =======================
    memory = Memory(location="cache/sklearn", verbose=10)
    pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineering(n_months=6)),  # Add feature engineering step
        ("imputer", SimpleImputer(strategy="median")),            # Handle missing values
        ("scaler", StandardScaler()),                             # Scale features
        ("model", regressor)                                      # Add the model
    ], memory=memory)  # Cache intermediate results to speed up GridSearchCV

    logger.info("Pipeline steps:")
    for name, step in pipeline.steps:
        logger.info(f"  {name:20}: {step.__class__.__name__}")

    # ======================= Grid Search =======================
    logger.info("Performing GridSearchCV...")

    metric = config["selection"]["primary_metric"] 
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grid, 
                               cv=config["gridCV"]["cv"], 
                               n_jobs=config["gridCV"]["n_jobs"], 
                               verbose=config["gridCV"]["verbose"],
                               scoring=config["metrics"],
                               refit=metric)
    
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
        "grid_search": grid_search
    }