# Customer Churn Prediction (MLOps Project)

![CI](https://github.com/14Hariom/churn-mlops/actions/workflows/ci.yml/badge.svg)

A machine learning project that predicts the **probability that a telecom customer will leave** (churn),
served through a REST API and tested automatically with GitHub Actions.

## The problem

A telecom company loses money when customers leave. Given a customer's details (contract type, tenure,
monthly bill, services), the model predicts the chance that the customer will churn, so the company can act early.

## What is inside

- `src/` : data cleaning, sample data generator, model training
- `app/main.py` : FastAPI service with a `/predict` endpoint
- `tests/` : pytest tests for data, model and API
- `.github/workflows/ci.yml` : runs the tests automatically on every push

## How to run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.generate_sample_data
python -m src.train
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs and try **POST /predict**.

## Run the tests

```bash
pytest -v
```

## Models compared

Logistic Regression, Random Forest and XGBoost. The best one (by F1 score) is saved automatically.

## Note about the data

This repo currently uses a synthetic demo dataset with the same columns as the Kaggle
Telco Customer Churn dataset. Results on demo data are not meaningful.

## Tech stack

Python, pandas, scikit-learn, XGBoost, FastAPI, Pydantic, pytest, GitHub Actions.
