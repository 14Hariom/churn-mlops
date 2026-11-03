from fastapi.testclient import TestClient

from app.main import app
from tests.test_model import LOYAL, RISKY

client = TestClient(app)


def test_home():
    r = client.get("/")
    assert r.status_code == 200


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_probability():
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert isinstance(body["will_churn"], bool)


def test_risky_customer_scores_higher_through_the_api():
    low = client.post("/predict", json=LOYAL).json()["churn_probability"]
    high = client.post("/predict", json=RISKY).json()["churn_probability"]
    assert high > low


def test_invalid_input_is_rejected():
    bad = {**RISKY, "Contract": "Three years"}     # not an allowed value
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_is_rejected():
    bad = {k: v for k, v in RISKY.items() if k != "tenure"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_prediction():
    r = client.post("/predict-batch", json=[LOYAL, RISKY])
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "best_model" in r.json()
