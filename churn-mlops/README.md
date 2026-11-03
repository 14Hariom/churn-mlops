# Customer Churn Prediction: End-to-End ML Project (MLOps)

![CI](https://github.com/YOUR-USERNAME/churn-mlops/actions/workflows/ci.yml/badge.svg)

A complete machine-learning project that goes **beyond a notebook**: it trains and compares models,
tracks experiments, serves predictions through a REST API, is tested automatically, runs in Docker,
deploys online, and can check for data drift.

> **Live demo:** `https://YOUR-APP.onrender.com/docs` *(add your link after deploying)*

## The problem

A telecom company loses money when customers leave ("churn"). Given a customer's details
(contract type, tenure, monthly bill, services...), predict the **probability that the customer will churn**,
so the company can act before it happens.

## How it fits together

```
 data/telco_churn.csv
        |
        v
  src/train.py  --(logs every run)-->  MLflow (parameters, metrics, models)
        |
        v
  models/model.pkl  (preprocessing + best model in one Pipeline)
        |
        v
  app/main.py  (FastAPI)  <-- pytest tests  <-- GitHub Actions (CI)
        |
        v
  Docker image  -->  Render / Hugging Face Spaces  -->  live API
        |
        v
  logs/predictions.csv  -->  src/drift.py  -->  drift report
```

## Project structure

```
churn-mlops/
|-- data/                    # dataset (telco_churn.csv)
|-- notebooks/01_eda.ipynb   # exploratory data analysis
|-- src/
|   |-- data.py              # column names + cleaning (shared by everything)
|   |-- generate_sample_data.py  # creates the demo dataset
|   |-- train.py             # train, compare, track (MLflow), save best model
|   `-- drift.py             # data drift check + HTML report
|-- app/main.py              # FastAPI service
|-- tests/                   # pytest tests (data, model, API)
|-- models/                  # trained model is saved here (git-ignored)
|-- .github/workflows/ci.yml # automatic tests + Docker build on every push
|-- Dockerfile
|-- render.yaml              # one-click deploy config for Render
|-- requirements.txt         # runtime dependencies
`-- requirements-dev.txt     # + MLflow, pytest, Jupyter
```

## About the dataset (important)

The repo ships with a **synthetic demo dataset** that has exactly the same columns as the Kaggle
*Telco Customer Churn* file. This lets you run everything immediately.

**For your portfolio, use the real data:**
1. Download **Telco Customer Churn** from Kaggle (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. Rename it to `data/telco_churn.csv` (replace the demo file).
3. Re-run `python -m src.train` and commit the new CSV.

Results on the demo data are not meaningful. Always report results from the real dataset.

## Quick start

```bash
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Mac / Linux

# 2. install everything
pip install -r requirements-dev.txt

# 3. train the models (creates models/model.pkl)
python -m src.train

# 4. start the API
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs**. Click *POST /predict* -> *Try it out* -> *Execute*.

Or from the terminal:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_request.json
```

Example response:

```json
{"churn_probability": 0.87, "will_churn": true, "risk_level": "High"}
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Is the service alive and is the model loaded? |
| `GET /model-info` | Which model won and its test metrics |
| `POST /predict` | Churn probability for one customer |
| `POST /predict-batch` | Same, for a list of customers |

## Experiment tracking with MLflow

Every training run logs its parameters, metrics and model.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open **http://localhost:5000** and compare the runs. *(Add a screenshot here.)*

## Tests

```bash
pytest -v
```

Tests cover data cleaning, the model (a risky customer must score higher than a loyal one),
and the API (valid input, invalid input, batch, health). They run automatically on every push
through GitHub Actions.

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

The model is trained **while the image is built**, so it always matches the library versions inside the image.

## Deploy (free) on Render

1. Push this project to GitHub.
2. On [render.com](https://render.com): **New +** -> **Web Service** -> connect your repo.
3. Choose **Docker** as the runtime and the **Free** plan (or use *Blueprint* so `render.yaml` does it for you).
4. Deploy. Your API will be at `https://<your-service>.onrender.com/docs`.

> The free tier sleeps when idle, so the first request can take 30-60 seconds.

**Hugging Face Spaces** also works: create a *Docker* Space, add `app_port: 8000` to the README
front-matter, and push the same files.

## Data drift monitoring

The API saves every request to `logs/predictions.csv`. Compare recent data with the training data:

```bash
python -m src.drift --current logs/predictions.csv   # real traffic
python -m src.drift --simulate                        # practice with fake drift
```

This writes `reports/drift_report.html` showing which features changed
(KS test for numbers, chi-square for categories). *(Add a screenshot here.)*

## Results

Fill this in after training on the real dataset (`models/metrics.json` has the numbers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

**Why F1/recall and not just accuracy?** Only about a quarter of customers churn, so a model that always
predicts "no churn" would be ~75% accurate but useless. Recall shows how many real churners we catch.

## Design decisions

- **One Pipeline** (preprocessing + model) is saved together, so the API cannot preprocess differently from training.
- **Baseline first:** Logistic Regression is compared against tree models before choosing a winner.
- **Class weights** handle the imbalance between churners and non-churners.
- **Pydantic validation** rejects bad requests (e.g. an unknown contract type) with a clear 422 error.
- **Small runtime image:** MLflow and Jupyter are dev-only dependencies, not in the Docker image.

## Limitations and future work

- The default threshold is 0.5; the best threshold depends on the cost of missing a churner vs. wasting a retention offer.
- Add SHAP explanations to show *why* a customer is at risk.
- Automate retraining when drift is detected.
- Add authentication and rate limiting to the API.
- Use a proper model registry (MLflow Model Registry) for staging/production versions.

## Tech stack

Python, pandas, scikit-learn, XGBoost, MLflow, FastAPI, Pydantic, pytest, Docker, GitHub Actions, Render.
