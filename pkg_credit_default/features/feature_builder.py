from pkg_credit_default.utils.logger import logger
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Pipeline for creating engineered credit risk features.
    """

    def __init__(self,n_months=6):
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
        X = self._calc_average_balance(X)
        X = self._calc_credit_utilization(X)
        X = self._calc_late_payment_M1(X)

        return X
    
    def _calc_average_balance(self, X):
        logger.info("Calculating average balance over the last {} months...".format(self.n_months))

        balance_vars = pd.DataFrame(columns=[f"balance_{i}" for i in range(1, self.n_months + 1)]) 
        for i in range(1, self.n_months + 1):          
            balance_vars[f"balance_{i}"] = X[f'BILL_AMT{i}'] - X[f'PAY_AMT{i}']  # Calculate balance for each month
        X['AVG_BALANCE_'] = balance_vars.mean(axis=1)   

        return X
    
    def _calc_credit_utilization(self, X):
        logger.info("Calculating credit utilization...")
        
        # Define credit utilization ratio as average balance divided by credit limit (LIMIT_BAL)
        X['CREDIT_UTILIZATION_'] = X['AVG_BALANCE_'] / X['LIMIT_BAL']
        
        return X
        
    def _calc_late_payment_M1(self, X):
        logger.info("Calculating late payment indicator for M1...")
        
        # Define late payment indicator as 1 if the payment in M1 (PAY_1) is late, else 0
        X['LATE_PAYMENT_M1_'] = (X['PAY_1'] > 0).astype(int)
        
        return X 
        


if __name__ == "__main__":
    print("Hello, World!")
    df = pd.DataFrame({
        "BILL_AMT1": [100, 200],
        "PAY_AMT1": [20, 50],
        "LIMIT_BAL": [1000, 2000],
        "PAY_1": [1, 0],
        "target": [0, 1],
    })
    fe = FeatureEngineering(n_months=1)
    df_transformed = fe._calc_average_balance(df.drop(columns=["target"]))
    print(df_transformed.head())