"""FastAPI service that serves the churn model.

Run locally (from the project root):

    uvicorn app.main:app --reload

Then open http://localhost:8000/docs for an interactive page.
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
LOG_PATH = ROOT / "logs" / "predictions.csv"
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer. Field names match the training data columns exactly."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1, description="1 if the customer is a senior citizen")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "Yes", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5, "TotalCharges": 470.0,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["Low", "Medium", "High"]


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Send customer details, get the probability that the customer will leave.",
    version="1.0.0",
)

_model = None


def get_model():
    """Load the model once, then reuse it for every request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model file not found. Run `python -m src.train` first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"


def log_prediction(features: dict, prob: float) -> None:
    """Save every request to logs/predictions.csv so we can check for drift later.

    Set the environment variable LOG_PREDICTIONS=0 to switch this off.
    """
    if os.getenv("LOG_PREDICTIONS", "1") != "1":
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **features, "churn_probability": round(prob, 4)}
        is_new = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass  # logging must never break a prediction


def make_prediction(model, customer: Customer) -> Prediction:
    features = customer.model_dump()
    prob = float(model.predict_proba(pd.DataFrame([features]))[0][1])
    log_prediction(features, prob)
    return Prediction(
        churn_probability=round(prob, 4),
        will_churn=prob >= THRESHOLD,
        risk_level=risk_level(prob),
    )


@app.get("/")
def home():
    return {"message": "Churn API is running. See /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found. Train the model first.")
    return json.loads(METRICS_PATH.read_text())


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return make_prediction(get_model(), customer)


@app.post("/predict-batch", response_model=List[Prediction])
def predict_batch(customers: List[Customer]):
    model = get_model()
    return [make_prediction(model, c) for c in customers]
