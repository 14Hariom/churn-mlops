# Customer Churn Prediction (MLOps Project)

![CI](https://github.com/14Hariom/churn-mlops/actions/workflows/ci.yml/badge.svg)

> **Live demo:** https://churn-mlops-g7bd.onrender.com/docs *(free tier: the first request can take 30-60 seconds to wake up)*

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

The data is the Kaggle Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv), 7,043 customers.

## Tech stack

Python, pandas, scikit-learn, XGBoost, FastAPI, Pydantic, pytest, GitHub Actions.


## Results (real Telco Churn dataset, 7,043 customers)

Evaluated on a 20% held-out test set (1,409 customers).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.783 | 0.614 | 0.842 |
| **Random Forest (best by F1)** | 0.757 | 0.529 | 0.781 | 0.631 | 0.843 |
| XGBoost | 0.750 | 0.520 | 0.778 | 0.623 | 0.840 |

**Why F1 and recall, not just accuracy?** About 27% of customers churn, so a model that always predicts
"no churn" would be about 73% accurate but catch no one. Recall shows how many real churners the model finds.

**Takeaway:** the three models perform very similarly. The simple Logistic Regression baseline is almost as
good as the tree models, which suggests the signal in this data is fairly simple (contract type, tenure, and so on).
