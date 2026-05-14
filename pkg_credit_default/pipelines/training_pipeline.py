# =================== IMPORTS =================== 
from pkg_credit_default.config.config_loader    import load_config
from pkg_credit_default.data.raw_loader         import load_data_from_csv
from pkg_credit_default.data.data_cleaning      import clean_data, remove_variables
from pkg_credit_default.data.data_preparation   import prepare_data
from pkg_credit_default.modeling.training_model import train_model
from pkg_credit_default.utils.logger            import logger
   
from sklearn.model_selection import train_test_split

# ========================================================= 
# =================== TRAINING PIPELINE =================== 
# ========================================================= 

def run_training_pipeline(model_type = ['logistic_regression'], save_output = True):

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
        best_models[model] = train_model(X_train, y_train, config, model_type=model, save_output=save_output)

    # PHASE 3 - Evaluate the model on the test set (if needed)


    # PHASE 4 - Select the champion model and save it (if needed)
    
    
####################### LAUNCH THE TRAINING PIPELINE ############################    

if __name__ == "__main__":
    #list_of_models = ["logistic_regression", "random_forest", "xgb_regressor", "knn", "svm", "lightgbm"]
    list_of_models = ["knn", "svm", "lightgbm"]
    
    run_training_pipeline(model_type=list_of_models)   