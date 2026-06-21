import pandas as pd
from pkg_credit_default.features.feature_builder import FeatureEngineering

# Create sample data
data = {
    'BILL_AMT1': [1000, 2000],
    'BILL_AMT2': [1200, 1800],
    'BILL_AMT3': [1100, 1900],
    'PAY_AMT1': [500, 600],
    'PAY_AMT2': [400, 700],
    'PAY_AMT3': [300, 800],
    'LIMIT_BAL': [5000, 8000],
    'PAY_1': [2, -1]
}

X_test = pd.DataFrame(data)

# Instantiate the class
fe = FeatureEngineering(n_months=3)

# Test public method
X_transformed = fe.transform(X_test)
print(X_transformed.head())

# Test private methods directly (perfectly fine for testing)
X_copy = X_test.copy()
X_copy = fe._calc_average_balance(X_copy)
X_copy = fe._calc_credit_utilization(X_copy)
X_copy = fe._calc_late_payment_M1(X_copy)

print(X_copy.columns)
print(X_copy[['AVG_BALANCE_', 'CREDIT_UTILIZATION_', 'LATE_PAYMENT_M1_']].head())