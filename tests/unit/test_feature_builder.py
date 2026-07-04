import pandas as pd
import pytest

from pkg_credit_default.features.feature_builder import FeatureEngineering


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "BILL_AMT1": [100, 200, 250, 300, 400, 500],
        "PAY_AMT1": [20, 50, 60, 70, 80, 90],
        "LIMIT_BAL": [1000, 2000, 2500, 3000, 4000, 5000],
        "PAY_1": [1, 0, 1, 0, 1, 0],
        "target": [0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def transformed_df(sample_df):
    fe = FeatureEngineering(n_months=1)
    X = sample_df.drop(columns=["target"])
    return fe.fit_transform(X)


# -------------------------
# Tests
# -------------------------

def test_engineered_columns_exist(transformed_df):
    """Check that all engineered columns are created."""
    for col in [
        "AVG_BALANCE_",
        "CREDIT_UTILIZATION_",
        "LATE_PAYMENT_M1_",
    ]:
        assert col in transformed_df.columns


def test_feature_names_contain_engineered_features(sample_df):
    """Check that get_feature_names_out includes new features."""
    fe = FeatureEngineering(n_months=1)
    X = sample_df.drop(columns=["target"])

    feature_names = fe.get_feature_names_out(X.columns.tolist())

    assert "AVG_BALANCE_" in feature_names
    assert "CREDIT_UTILIZATION_" in feature_names
    assert "LATE_PAYMENT_M1_" in feature_names


def test_feature_names_match_output(transformed_df, sample_df):
    """Ensure declared feature names match transformed DataFrame."""
    fe = FeatureEngineering(n_months=1)

    X = sample_df.drop(columns=["target"])
    feature_names = fe.get_feature_names_out(X.columns.tolist())

    assert feature_names == transformed_df.columns.tolist()


def test_avg_balance_calculation(transformed_df):
    """Validate correct AVG_BALANCE_ computation."""
    expected = [
        100 - 20,
        200 - 50,
        250 - 60,
        300 - 70,
        400 - 80,
        500 - 90,
    ]

    assert list(transformed_df["AVG_BALANCE_"]) == expected