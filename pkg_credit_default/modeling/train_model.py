from datetime import datetime
import os 
import joblib

from pkg_credit_default.utils.logger import logger
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor   
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


def train_model(X, y, config, model_type="logistic_regression" ):

    logger.info(f"Training {model_type} model...")

    model_type = model_type.lower()

    if model_type == "logistic_regression":
        regressor = LogisticRegression
        param_grid = {}

    else:
        print("no model")

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
        ("scaler", StandardScaler()),  # Scale features
        ("feature_engineering", FeatureEngineering(n_months=6)),  # Add feature engineering step
        ("model", regressor())  # Add the model
    ])

    logger.info("Performing grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    logger.info("Model training completed.")

    best_model = grid_search.best_estimator_
    
    avg_cv_score = grid_search.best_score_
    logger.info(f"Best CV Score: {avg_cv_score:.4f}")
    
    std_cv_score = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    logger.info(f"CV Score Std Dev: {std_cv_score:.4f}")
    
    train_score = best_model.score(X, y)
    logger.info(f"Training Score: {train_score:.4f}")
    
    logger.info(f"Best Hyperparameters: {grid_search.best_params_}")
      
    # store the best model and its performance metrics for later use
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    
    model_dir = os.path.join(config["model"]["output_dir"], timestamp)
    os.makedirs(model_dir, exist_ok=True)                    # create folder if it doesn't exist

    model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")
    logger.info(f"Timestamp: {timestamp}")

    return best_model, avg_cv_score, std_cv_score, train_score, grid_search.best_params_