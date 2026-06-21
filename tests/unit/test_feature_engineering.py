import pandas as pd

from sklearn.pipeline import Pipeline

from pkg_credit_default.features.feature_builder import FeatureEngineering

def create_sample_dataframe():

    return pd.DataFrame({
        "BILL_AMT1": [100, 200],
        "PAY_AMT1": [20, 50],
        "LIMIT_BAL": [1000, 2000],
        "PAY_1": [1, 0],
        "target": [0, 1],
    })


def test_fit_returns_self():

    # Arrange
    df = create_sample_dataframe()
    X = df.drop(columns=["target"])
    transformer = FeatureEngineering(n_months=1)

    # Act
    result = transformer.fit(X)

    # Assert
    assert result is transformer


def test_transform_creates_expected_columns():

    # Arrange
    df = create_sample_dataframe()
    X = df.drop(columns=["target"])
    transformer = FeatureEngineering(n_months=1)

    # Act
    result = transformer.fit_transform(X)

    # Assert
    expected_columns = [
        "AVG_BALANCE_",
        "CREDIT_UTILIZATION_",
        "LATE_PAYMENT_M1_",
    ]

    for col in expected_columns:
        assert col in result.columns


def test_transform_does_not_modify_original_dataframe():

    # Arrange
    df = create_sample_dataframe()
    X = df.drop(columns=["target"])
    original_columns = X.columns.tolist()
    transformer = FeatureEngineering(n_months=1)

    # Act
    _ = transformer.fit_transform(X)

    # Assert
    assert X.columns.tolist() == original_columns


def test_avg_balance_calculation():

    # Arrange
    df = create_sample_dataframe()

    X = df.drop(columns=["target"])

    transformer = FeatureEngineering(n_months=1)

    # Act
    result = transformer.fit_transform(X)

    # Assert
    expected_balance_row_0 = 100.0 - 20.0
    expected_balance_row_1 = 200.0 - 50.0

    assert result.loc[0, "AVG_BALANCE_"] == expected_balance_row_0
    assert result.loc[1, "AVG_BALANCE_"] == expected_balance_row_1


""" def test_pipeline_integration():

    # Arrange
    df = create_sample_dataframe()

    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = Pipeline([
        ("features", FeatureEngineering(n_months=1)),
        ("model", RandomForestClassifier(random_state=42)),
    ])

    # Act
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    # Assert
    assert len(predictions) == len(X) 
"""