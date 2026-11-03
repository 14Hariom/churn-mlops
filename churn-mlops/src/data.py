"""Shared constants and data-cleaning code.

This file is imported by training (src/train.py), the API (app/main.py),
the drift check (src/drift.py) and the tests, so everybody uses the SAME
column names and the SAME cleaning rules.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"

TARGET = "Churn"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV. If it does not exist, create the demo dataset first."""
    path = Path(path)
    if not path.exists():
        from src.generate_sample_data import generate

        print(f"[data] {path} not found -> generating a synthetic demo dataset.")
        generate(path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known problems of the Telco churn dataset.

    - customerID is only an identifier, so we drop it
    - TotalCharges is text in the raw file and contains blanks -> make it numeric
    - text columns may have stray spaces -> strip them
    - Churn "Yes"/"No" -> 1/0
    """
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    if TARGET in df.columns:
        mapped = df[TARGET].astype(str).str.strip().map(
            {"Yes": 1, "No": 0, "1": 1, "0": 0}
        )
        if mapped.isna().any():
            raise ValueError("Churn column contains values other than Yes/No.")
        df[TARGET] = mapped.astype(int)

    return df
