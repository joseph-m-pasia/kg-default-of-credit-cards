from pkg_credit_default.config.config_loader import load_config
from pkg_credit_default.data.loaders.raw_loader import load_data_from_csv
from pkg_credit_default.utils.logger import logger

def run_training():

    logger.info("run_training() - Starting training pipeline...")

    # STEP 1: CONFIG
    config = load_config()

    print(config)

    # STEP 2: RAW DATA
    df = load_data_from_csv(config["data"]["raw_path"])

    # STEP 3: FEATURES
    # X, y = build_features(df, config)

    # STEP 4: MODEL TRAINING
    # model = train_model(X, y, config)

    model = config
    return model

run_training()