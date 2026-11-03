import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (CATEGORICAL_FEATURES, DATA_PATH, FEATURES, MODELS_DIR, NUMERIC_FEATURES,
                      ROOT, TARGET, clean_data, load_raw_data)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def get_candidates(y_train: pd.Series) -> list:
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    candidates = [
        ("logistic_regression",
         LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
         {"max_iter": 1000, "class_weight": "balanced"}),
        ("random_forest",
         RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                class_weight="balanced_subsample", n_jobs=-1,
                                random_state=RANDOM_STATE),
         {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5,
          "class_weight": "balanced_subsample"}),
    ]

    if XGB_AVAILABLE:
        candidates.append((
            "xgboost",
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, scale_pos_weight=pos_weight,
                          eval_metric="logloss", random_state=RANDOM_STATE),
            {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
             "scale_pos_weight": round(pos_weight, 2)},
        ))
    else:
        candidates.append((
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                           class_weight="balanced", random_state=RANDOM_STATE),
            {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300,
             "class_weight": "balanced"},
        ))
    return candidates


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def run_training(use_mlflow: bool = True, data_path: Path = DATA_PATH,
                 models_dir: Path = MODELS_DIR) -> dict:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_raw_data(data_path))
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"[train] rows: {len(df)} | train: {len(X_train)} | test: {len(X_test)} "
          f"| churn rate: {y.mean():.1%}")

    tracking = use_mlflow and MLFLOW_AVAILABLE
    if use_mlflow and not MLFLOW_AVAILABLE:
        print("[train] mlflow not installed -> skipping experiment tracking.")
    if tracking:
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("churn-prediction")

    results, fitted = {}, {}
    for name, estimator, params in get_candidates(y_train):
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name], fitted[name] = metrics, pipe
        print(f"[train] {name:<24} " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        if tracking:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                try:
                    mlflow.sklearn.log_model(pipe, "model")
                except Exception as exc:
                    print(f"[train] warning: could not log model artifact to MLflow: {exc}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"[train] best model by F1: {best_name}")

    joblib.dump(fitted[best_name], models_dir / "model.pkl")
    X_train.sample(n=min(1000, len(X_train)), random_state=RANDOM_STATE) \
        .to_csv(models_dir / "reference_data.csv", index=False)

    summary = {
        "best_model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "features": FEATURES,
        "results": results,
    }
    (models_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] saved model to {models_dir / 'model.pkl'}")
    return summary


if __name__ == "__main__":
    run_training(use_mlflow="--no-mlflow" not in sys.argv)
    