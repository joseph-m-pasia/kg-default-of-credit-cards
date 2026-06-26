# from pkg_credit_default.utils.logger import logger
# from pkg_credit_default.pipelines.training_pipeline import run_training_pipeline    


    
#==================== LAUNCH THE TRAINING PIPELINE ==============================    
# list_of_models = ["logistic_regression", "random_forest", "xgb_regressor", "knn", "lightgbm"]

# champion_model = run_training_pipeline(model_type=list_of_models, params={"save_output": True, "plot_metrics": False, "save_split_data": True})               

# logger.info(f"Champion model based on F1 score: {champion_model['champion_model'][0]} with F1 score = {champion_model['champion_model'][1]['best_score']:.4f}") 

#==================== COPY AND RENAME THE CHAMPION MODEL ==============================    
import shutil


champion_model_dir = "artifacts/models/20260626_142823_champion_logistic_regression/"  

shutil.copy(f"{champion_model_dir}/champion_logistic_regression_model.pkl", "artifacts/model.pkl")
