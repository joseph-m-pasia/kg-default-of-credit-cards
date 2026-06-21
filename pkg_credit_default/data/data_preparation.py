from pkg_credit_default.utils.logger import logger

from pkg_credit_default.data.raw_loader import load_data_from_csv
from pkg_credit_default.data.data_cleaning import clean_data, remove_variables


def prepare_data(config):

    logger.info("Preparing data for modeling...")

    # STEP 1: RAW DATA
    df = load_data_from_csv(
        config["data"]["raw_path"], config["data"]["features"] + [config["data"]["target_variable"]]
    )

    # STEP 2: DATA CLEANING
    df = clean_data(df)

    # STEP 3: REMOVE UNNCESSARY VARIABLES
    df = remove_variables(df, config)

    # STEP 4: SPLIT INTO X AND y
    X = df.drop(config["data"]["target_variable"], axis=1)
    y = df[config["data"]["target_variable"]]

    return X, y
