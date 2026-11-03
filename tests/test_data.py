import pandas as pd

from src.data import CATEGORICAL_FEATURES, FEATURES, clean_data


def make_raw_row(**overrides):
    row = {c: "No" for c in CATEGORICAL_FEATURES}
    row.update({"customerID": "0001-ABCDE", "SeniorCitizen": 0, "tenure": 0,
                "MonthlyCharges": 29.85, "TotalCharges": " ", "Churn": "Yes"})
    row.update(overrides)
    return row


def test_clean_data_fixes_known_problems():
    df = clean_data(pd.DataFrame([make_raw_row()]))
    assert "customerID" not in df.columns
    assert df.loc[0, "TotalCharges"] == 0.0
    assert df.loc[0, "Churn"] == 1


def test_clean_data_maps_no_to_zero():
    df = clean_data(pd.DataFrame([make_raw_row(Churn="No", TotalCharges="120.5")]))
    assert df.loc[0, "Churn"] == 0
    assert df.loc[0, "TotalCharges"] == 120.5


def test_all_feature_columns_exist_after_cleaning():
    df = clean_data(pd.DataFrame([make_raw_row()]))
    assert all(col in df.columns for col in FEATURES)
    