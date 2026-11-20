"""Train the initial housing-price champion on the historical portion of the supplied data."""
from pathlib import Path
import json
import joblib
import pandas as pd
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Housing-selected-columns.csv"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "price"
RANDOM_STATE = 42

def build_pipeline(params=None):
    df = pd.read_csv(DATA_PATH)
    features = [c for c in df.columns if c != TARGET]
    categorical = df[features].select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in features if c not in categorical]
    preprocess = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model_params = params or {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2,
                              "random_state": RANDOM_STATE, "n_jobs": -1}
    return Pipeline([("preprocessor", preprocess),
                     ("regressor", RandomForestRegressor(**model_params))])

def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    # Stable simulation: 70% historical training, 15% incoming labelled batch,
    # and 15% untouched holdout for an apples-to-apples quality comparison.
    X_hist, X_rest, y_hist, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_new, X_holdout, y_new, y_holdout = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=RANDOM_STATE)

    model = build_pipeline()
    model.fit(X_hist, y_hist)
    pred = model.predict(X_holdout)
    metrics = {
        "mae": float(mean_absolute_error(y_holdout, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_holdout, pred))),
        "r2": float(r2_score(y_holdout, pred)),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "housing_champion.joblib")
    # Save deterministic split so monitoring can reproduce exactly the same evaluation.
    pd.concat([X_hist, y_hist], axis=1).to_csv(ARTIFACT_DIR / "historical_training_data.csv", index=False)
    pd.concat([X_new, y_new], axis=1).to_csv(ARTIFACT_DIR / "incoming_labelled_batch.csv", index=False)
    pd.concat([X_holdout, y_holdout], axis=1).to_csv(ARTIFACT_DIR / "quality_holdout.csv", index=False)
    report = {"model": "RandomForestRegressor", "role": "initial_champion",
              "rows": int(len(X_hist)), "holdout_rows": int(len(X_holdout)), **metrics}
    (ARTIFACT_DIR / "initial_training_report.json").write_text(json.dumps(report, indent=2))
    try:
        if mlflow is None:
            raise RuntimeError('MLflow is not installed; install requirements.txt to enable tracking')
        mlflow.set_tracking_uri("file:" + str(ROOT / "mlruns"))
        mlflow.set_experiment("housing-price-monitoring")
        with mlflow.start_run(run_name="initial-champion") as run:
            mlflow.log_params({"model": "RandomForestRegressor", "n_estimators": 200,
                               "max_depth": 12, "training_rows": len(X_hist)})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(MODEL_DIR / "housing_champion.joblib"))
    except Exception as exc:
        print(f"MLflow tracking skipped: {exc}")
    print("Initial champion saved:", MODEL_DIR / "housing_champion.joblib")
    print("Initial metrics:", json.dumps(metrics, indent=2))
    print(f"Split sizes: historical={len(X_hist)}, incoming={len(X_new)}, holdout={len(X_holdout)}")

if __name__ == "__main__":
    main()
