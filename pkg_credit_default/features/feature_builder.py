from pkg_credit_default.utils.logger import logger
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Pipeline for creating engineered credit risk features.
    """

    def __init__(self, n_months=6):
        self.n_months = n_months

    def fit(self, X, y=None):
        """
        Fit the feature engineering pipeline.
        """

        # No fitting required for rule-based features   
        return self

    def transform(self, X):
        """
        Transform the data using the feature engineering pipeline.
        """
        X = X.copy()  # Avoid modifying original dataframe
       
        balance_vars = pd.DataFrame({
            f"balance_{i}": X[f"BILL_AMT{i}"] - X[f"PAY_AMT{i}"]
            for i in range(1, self.n_months + 1)
        })

        X["AVG_BALANCE_"] = balance_vars.mean(axis=1)
        X["CREDIT_UTILIZATION_"] = X["AVG_BALANCE_"] / X["LIMIT_BAL"].replace(0, np.nan)
        pay_col = "PAY_0" if "PAY_0" in X.columns else "PAY_1"
        X["LATE_PAYMENT_M1_"] = (X[pay_col] > 0).astype(int)

        return X

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            raise ValueError("input_features must be provided.")

        return list(input_features) + [
            "AVG_BALANCE_",
            "CREDIT_UTILIZATION_",
            "LATE_PAYMENT_M1_"
        ]

####################### EXAMPLE USAGE ############################

if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "BILL_AMT1": [100, 200],
            "PAY_AMT1": [20, 50],
            "LIMIT_BAL": [1000, 2000],
            "PAY_1": [1, 0],
            "target": [0, 1],
        }
    )
    fe = FeatureEngineering(n_months=1)
    df_transformed = fe.transform(df.drop(columns=["target"]))
    print(df_transformed)
