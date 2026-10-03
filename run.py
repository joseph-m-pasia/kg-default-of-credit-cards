from pkg_house_prices.utils.logger                  import logger
from pkg_house_prices.utils.config                  import CONFIG
from pkg_house_prices.data.data_loader              import X_train, y_train
from pkg_house_prices.models.task_01_train_models   import train_model

    
#==================== LAUNCH THE TRAINING PIPELINE ==============================    git s


# show me the variables in X_train with NaN values
nan_columns = X_train.columns[X_train.isna().any()].tolist() 
logger.info(f"Columns in X_train with NaN values: {nan_columns}")

# Train the model
lr_model,    lr_cv_score, lr_train_score, lr_std_cv_score = train_model(X_train, y_train, "linear")
lasso_model, lasso_cv_score, lasso_train_score, lasso_std_cv_score = train_model(X_train, y_train, "lasso")
ridge_model, ridge_cv_score, ridge_train_score, ridge_std_cv_score = train_model(X_train, y_train, "ridge")
enet_model,  elasticnet_cv_score, elasticnet_train_score, elasticnet_std_cv_score = train_model(X_train, y_train, "elasticnet")    
x_gb_model,  x_gb_cv_score, x_gb_train_score, x_gb_std_cv_score = train_model(X_train, y_train, "xgboost")  

# print CV scores, train scores, and std CV scores for all models and identify champion model
cv_scores = {
    "Linear Regression": lr_cv_score,
    "Lasso Regression": lasso_cv_score,
    "Ridge Regression": ridge_cv_score,
    "ElasticNet Regression": elasticnet_cv_score,
    "XGBoost Regression": x_gb_cv_score
}
    
train_scores = {
    "Linear Regression": lr_model.score(X_train, y_train) if lr_model else None,
    "Lasso Regression": lasso_train_score,
    "Ridge Regression": ridge_train_score,
    "ElasticNet Regression": elasticnet_train_score,
    "XGBoost Regression": x_gb_train_score
}       
    
std_cv_scores = {
    "Linear Regression": lr_std_cv_score,
    "Lasso Regression": lasso_std_cv_score,
    "Ridge Regression": ridge_std_cv_score,         
    "ElasticNet Regression": elasticnet_std_cv_score,
    "XGBoost Regression": x_gb_std_cv_score
}

for model_name, cv_score in cv_scores.items():
    logger.info(f"{model_name} - CV R^2 Score: {cv_score:.4f}")     
    logger.info(f"{model_name} - Train R^2 Score: {train_scores[model_name]:.4f}")
    logger.info(f"{model_name} - CV R^2 Score Std Dev: {std_cv_scores[model_name]:.4f}")    

champion_model = max(cv_scores, key=cv_scores.get)  
logger.info(f"Champion model based on CV R^2 score: {champion_model} with score {cv_scores[champion_model]:.4f}, Train R^2 score: {train_scores[champion_model]:.4f}, CV R^2 score std dev: {std_cv_scores[champion_model]:.4f}")   

# print the champion model's hyperparameters (if applicable)
if champion_model != "Linear Regression":
    champion_model_pipeline = {
        "Lasso Regression": lasso_model,
        "Ridge Regression": ridge_model,
        "ElasticNet Regression": enet_model,
        "XGBoost Regression": x_gb_model
    }[champion_model]
    logger.info(f"Champion model hyperparameters: {champion_model_pipeline.named_steps['regressor']}")  

#==================== COPY AND RENAME THE CHAMPION MODEL ==============================    
import shutil

#champion_model_dir = "artifacts/models/20260704_184917_champion_logistic_regression/"  



#shutil.copy(f"{champion_model_dir}/champion_logistic_regression_model.pkl", "artifacts/model.pkl")
            
