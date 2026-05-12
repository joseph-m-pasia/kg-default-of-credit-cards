from datetime import datetime
import os
from pyexpat import model 
import joblib

from pkg_credit_default.utils.logger import logger
from pkg_credit_default.features.feature_builder import FeatureEngineering

from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor   
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train, config, model_type="logistic_regression" ):

    logger.info(f"Training {model_type} model...")

    model_type = model_type.lower()

    # ======================= Model Selection =======================
    if model_type == "logistic_regression":
        regressor = LogisticRegression
        param_grid = config["models"]["logistic_regression"]["param_grid"]
    elif model_type == "xgb_regressor":
        regressor = XGBRegressor
        param_grid = config["models"]["xgb_regressor"]["param_grid"]
    elif model_type == "random_forest":
        regressor = RandomForestRegressor
        param_grid = config["models"]["random_forest"]["param_grid"]
    elif model_type == "svm":
        regressor = SVR
        param_grid = config["models"]["svm"]["param_grid"]
    elif model_type == "knn":
        regressor = KNeighborsRegressor
        param_grid = config["models"]["knn"]["param_grid"]
    elif model_type == "lightgbm":
        regressor = LGBMRegressor
        param_grid = config["models"]["lightgbm"]["param_grid"]
    else:
        raise ValueError(f"Model type '{model_type}' is not supported yet.")

    # ======================= Create Pipeline =======================
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
        ("scaler", StandardScaler()),  # Scale features
        ("feature_engineering", FeatureEngineering(n_months=6)),  # Add feature engineering step
        ("model", regressor())  # Add the model
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
                               verbose=config["gridCV"]["verbose"])
    grid_search.fit(X_train, y_train)
    logger.info("Model training completed.")

    best_model = grid_search.best_estimator_
    
    # ======================= Logging Results =======================
    logger.info(f"Best CV Score: {grid_search.best_score_:.4f}")
    logger.info(f"CV Score Std Dev: {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f}")
    logger.info(f"Training Score: {best_model.score(X_train, y_train):.4f}")
    logger.info(f"Best Parameters: {grid_search.best_params_}")
      
    # ==================== Save Model ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    
    model_dir = os.path.join(config["models"]["output_dir_models"], timestamp)
    os.makedirs(model_dir, exist_ok=True)                    # create folder if it doesn't exist

    model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
    joblib.dump(best_model, model_path)
  
    logger.info(f"Best model saved to {model_path}")
  
    return {
        "model": best_model,
        "best_score": grid_search.best_score_,
        "best_params": grid_search.best_params_,
        "model_dir": model_dir,
        "model_path": model_path,
        "grid_search": grid_search
    }