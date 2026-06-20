# =================== IMPORTS =================== 
from pkg_credit_default.config.config_loader    import load_config
from pkg_credit_default.data.raw_loader         import load_data_from_csv
from pkg_credit_default.data.data_cleaning      import clean_data, remove_variables
from pkg_credit_default.data.data_preparation   import prepare_data
from pkg_credit_default.modeling.trainer        import train_model
from pkg_credit_default.utils.logger            import logger
from pkg_credit_default.utils.utils             import save_model
from pkg_credit_default.modeling.evaluator      import evaluate_model
   
from sklearn.model_selection import train_test_split
import os


# ========================================================= 
# =================== TRAINING PIPELINE =================== 
# ========================================================= 
        

def run_training_pipeline(model_type: list = ['logistic_regression'], params: dict = {}):

    """
    Run the full training pipeline: data loading, cleaning, preparation, model training, evaluation, and saving.
    Input:
        model_type: list of model types to train (e.g. ["logistic_regression", "random_forest", "xgb_regressor", "knn", "svm", "lightgbm"])
        params:     dict of additional parameters (e.g. {"save_output": True, "plot_metrics": False, save_split_data: False})
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
    if params.get("save_split_data", False):
        # Save the split data
        logger.info("Saving split data...")
        train_df = X_train.copy()
        train_df["target"] = y_train
        test_df = X_test.copy()
        test_df["target"] = y_test

        # create directories if they don't exist
        os.makedirs(config["paths"]["output_dir_data"], exist_ok=True)
        train_df.to_csv(os.path.join(config["paths"]["output_dir_data"], "train.csv"), index=False)
        test_df.to_csv(os.path.join(config["paths"]["output_dir_data"], "test.csv"),   index=False)
    
    # STEP 3 - Build and train the model
    best_models = {} 
    for model in model_type:
        logger.info(f"Training {model}...")
        best_models[model] = train_model(X_train, 
                                         y_train, 
                                         config, 
                                         model_type=model, 
                                         save_output=save_output)

    # STEP 4 - Select the champion model and save it (if needed) based on the metric specified in GridSearchCV config
    logger.info(f"Selecting champion model based on {config['selection']['primary_metric']}...")
    best_model_name, best_model_info = max(best_models.items(),     
                                           key=lambda item: item[1]["best_score"])
    champion_model = (best_model_name, best_model_info)
    model_dir = config["paths"]["output_dir_models"]
    model_path = save_model(champion_model, model_dir, f"champion_{best_model_name}", timestamp=True)

    # STEP 5 - Evaluate the champion model using the test set (if needed)
    logger.info(f"Evaluating champion model {champion_model[0]} on test set...")
    test_metrics = evaluate_model(champion_model[1]["model"], X_test, y_test)
    logger.info(f"TEST SET evaluation metrics for champion model -> {champion_model[0]}: {test_metrics}")

    test_metrics = evaluate_model(champion_model[1]["model"], X_train, y_train)
    logger.info(f"TRAIN SET evaluation metrics for champion model -> {champion_model[0]}: {test_metrics}")

   
    logger.info("Training pipeline completed.")
    return {
        "best_models": best_models,   
        "champion_model": champion_model,
        "test_metrics": test_metrics
    }    
    
####################### EXAMPLE USAGE ############################    

if __name__ == "__main__":
    list_of_models = ["logistic_regression"] # "xgb_regressor", "knn", "lightgbm"]
    
    champion_model = run_training_pipeline(model_type = list_of_models, 
                                           params     = {"save_output": False, 
                                                         "plot_metrics": True, 
                                                         "save_split_data": True})               
    logger.info(f"Champion   model based on F1 score: {champion_model['champion_model'][0]} with F1 score = {champion_model['champion_model'][1]['best_score']:.4f}")