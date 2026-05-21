# =================== IMPORTS =================== 
from pkg_credit_default.config.config_loader    import load_config
from pkg_credit_default.data.raw_loader         import load_data_from_csv
from pkg_credit_default.data.data_cleaning      import clean_data, remove_variables
from pkg_credit_default.data.data_preparation   import prepare_data
from pkg_credit_default.modeling.trainer        import train_model
from pkg_credit_default.modeling.evaluator      import evaluate_model, select_champion_model
from pkg_credit_default.utils.logger            import logger
   
from sklearn.model_selection import train_test_split

# ========================================================= 
# =================== TRAINING PIPELINE =================== 
# ========================================================= 
        

def run_training_pipeline(model_type = ['logistic_regression'], params={}):

    """
    Run the full training pipeline: data loading, cleaning, preparation, model training, evaluation, and saving.
    Input:
        model_type: list of model types to train (e.g. ["logistic_regression", "random_forest", "xgb_regressor", "knn", "svm", "lightgbm"])
        params:     dict of additional parameters (e.g. {"save_output": True, "plot_metrics": False})
    """

    # initialize the parameters
    save_output = params.get("save_output", True)
    plot_metrics = params.get("plot_metrics", False)

    logger.info("Starting training pipeline...")

    # STEP 0 - Load the configuration
    logger.info("Loading configuration...")
    config = load_config()

    # STEP 1 - Prepare the data
    logger.info("Preparing data...")
    X, y = prepare_data(config)    

    # STEP 2 : Split into train and test sets (if needed)
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=config["data"]["test_size"], 
                                                        random_state=config["data"]["random_state"])
    
    # STEP 3 - Build and train the model
    best_models = {} 
    for model in model_type:
        logger.info(f"Training {model}...")
        best_models[model] = train_model(X_train, 
                                         y_train, 
                                         config, 
                                         model_type=model, 
                                         save_output=save_output)

    # STEP 4 - Evaluate the model on the test set (if needed)
    score_results = {}
    for model in model_type:
        logger.info(f"Evaluating {model}...")
        score_results[model] = evaluate_model(best_models[model]["model"], 
                                              X_test, 
                                              y_test)
    # STEP 5 - Select the champion model and save it (if needed) based on the metric specified in GridSearchCV config
    score_models = {model: scores[config["selection"]["primary_metric"]] for model, scores in score_results.items()}
    score_models = sorted(score_models.items(), key=lambda x: x[1], reverse=True)
    champion_model = select_champion_model(score_models, 
                                           metric=config["selection"]["primary_metric"], 
                                           plot_metrics=plot_metrics)

    logger.info("Training pipeline completed.")
    return {
        "best_models": best_models,   
        "score_results": score_results,
        "champion_model": champion_model
    }    
    
####################### EXAMPLE USAGE ############################    

if __name__ == "__main__":
    #list_of_models = ["logistic_regression", "random_forest", "xgb_regressor", "knn", "lightgbm"]
    list_of_models = ["logistic_regression","xgb_regressor"]
    
    champion_model = run_training_pipeline(model_type=list_of_models, params={"save_output": True, "plot_metrics": True})   
    logger.info(f"Champion model based on F1 score: {champion_model['champion_model'][0]} with F1 score = {champion_model['champion_model'][1]:.4f}")