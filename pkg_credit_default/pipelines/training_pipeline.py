from pkg_credit_default.config.loader import config_loader
from pkg_credit_default.data.loaders.raw_loader import raw_loaders

def run_training():
    # STEP 1: CONFIG
    config = load_config()

    # STEP 2: RAW DATA
    df = load_raw_data(config)

    # STEP 3: FEATURES
    # X, y = build_features(df, config)

    # STEP 4: MODEL TRAINING
    # model = train_model(X, y, config)

    return model