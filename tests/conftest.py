import os

import pytest

os.environ["LOG_PREDICTIONS"] = "0"


@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    from src.data import MODELS_DIR

    if not (MODELS_DIR / "model.pkl").exists():
        from src.train import run_training

        run_training(use_mlflow=False)