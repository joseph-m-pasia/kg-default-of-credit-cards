from pkg_credit_default.utils.logger import logger
from pkg_credit_default.pipelines.training_pipeline import run_training_pipeline    

    
#==================== LAUNCH THE TRAINING PIPELINE ==============================    
list_of_models = ["logistic_regression", "random_forest", "xgb_regressor", "knn", "lightgbm"]

run_training_pipeline(model_type=list_of_models)