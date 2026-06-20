from pkg_credit_default.utils.logger            import logger
from pkg_credit_default.config.config_loader    import load_config
from pkg_credit_default.utils.utils             import load_ml_model
from pkg_credit_default.modeling.evaluator      import evaluate_model
import os
import pandas as pd

    
if __name__ == "__main__":
    # load the ml (e.g. champion) model and its metrics
    config = load_config()
    model_path = config["paths"]["output_dir_models"] + "/20260602_111120_champion_logistic_regression/" + "champion_logistic_regression_model.pkl"
    champion_model = load_ml_model(model_path=model_path)
    print(f"Champion model: {champion_model}") 
    
    # evaluate the champion model on the test set (if needed)
    test_data_path = config["paths"]["output_dir_data"] + "/test.csv"
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    evaluate_model(champion_model[1]["model"], X_test, y_test)

    #evaluate the champion model on the training set (if needed)
    train_data_path = config["paths"]["output_dir_data"] + "/train.csv"
    train_df = pd.read_csv(train_data_path)
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    evaluate_model(champion_model[1]["model"], X_train, y_train)