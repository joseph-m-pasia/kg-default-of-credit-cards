from pkg_credit_default.config.config_loader import load_config
from pkg_credit_default.data.loaders.raw_loader import load_data_from_csv
from pkg_credit_default.utils.logger import logger
from pkg_credit_default.features.feature_builder import create_new_features
from pkg_credit_default.data.preprocessors.data_cleaning import clean_data, remove_variables
    
def run_training():

    logger.info("run_training() - Starting training pipeline...")

    # STEP 1: CONFIG
    config = load_config()

    # STEP 2: RAW DATA
    df = load_data_from_csv(config["data"]["raw_path"])

    # STEP 3: DATA CLEANING
    df = clean_data(df)

    # STEP 4: FEATURES
    df = create_new_features(df, config)

    # STEP 5: REMOVE UNNCESSARY VARIABLES
    df = remove_variables(df, config)

    print(df.columns)

    # STEP 6  : MODEL TRAINING
    # model = train_model(X, y, config)

    model = config
    return model

run_training()