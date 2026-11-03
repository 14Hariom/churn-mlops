import json

import joblib
import pandas as pd

from src.data import FEATURES, MODELS_DIR

LOYAL = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
    "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 60.0, "TotalCharges": 3600.0,
}
RISKY = {**LOYAL, "tenure": 1, "InternetService": "Fiber optic", "OnlineSecurity": "No",
         "TechSupport": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
         "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0}


def test_model_file_and_metrics_exist():
    assert (MODELS_DIR / "model.pkl").exists()
    summary = json.loads((MODELS_DIR / "metrics.json").read_text())
    assert 0 <= summary["results"][summary["best_model"]]["f1"] <= 1


def test_risky_customer_scores_higher_than_loyal_customer():
    model = joblib.load(MODELS_DIR / "model.pkl")
    df = pd.DataFrame([LOYAL, RISKY])[FEATURES]
    loyal_p, risky_p = model.predict_proba(df)[:, 1]
    assert risky_p > loyal_p
