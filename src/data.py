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
    path = Path(path)
    if not path.exists():
        from src.generate_sample_data import generate
        print(f"[data] {path} not found -> generating a synthetic demo dataset.")
        generate(path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
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