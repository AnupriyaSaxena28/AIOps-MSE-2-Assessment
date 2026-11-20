"""Validate incoming labelled housing data, evaluate champion, retrain, and apply a quality gate."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Housing-selected-columns.csv"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
TARGET = "price"
REQUIRED = ["price", "area", "bedrooms", "bathrooms", "stories", "mainroad",
            "guestroom", "basement", "hotwaterheating", "airconditioning"]
RANDOM_STATE = 42
# RMSE is the primary gate: it measures typical prediction error in the target's
# currency units and penalizes large misses. Candidate must improve by >=1%.
MIN_RMSE_IMPROVEMENT = 0.01

def metrics_for(model, frame):
    X, y = frame.drop(columns=[TARGET]), frame[TARGET]
    pred = model.predict(X)
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred))}

def make_candidate():
    sample = pd.read_csv(DATA)
    X = sample.drop(columns=[TARGET])
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    prep = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("preprocessor", prep),
        ("regressor", RandomForestRegressor(
            n_estimators=350, max_depth=None, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1))
    ])

def validate_batch(df):
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df[REQUIRED].isnull().any().any():
        counts = df[REQUIRED].isnull().sum()
        raise ValueError(f"Missing values detected: {counts[counts > 0].to_dict()}")
    if df.duplicated().any():
        raise ValueError(f"Duplicate records detected: {int(df.duplicated().sum())}")
    if (df[TARGET] <= 0).any():
        raise ValueError("Target 'price' must be positive")
    return {"rows": int(len(df)), "columns": int(len(df.columns)),
            "duplicate_records": 0, "missing_values": 0,
            "target_distribution": {str(k): int(v) for k, v in
                                    pd.cut(df[TARGET], bins=5).value_counts().sort_index().items()}}

def main():
    ARTIFACTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    incoming = pd.read_csv(ARTIFACTS / "incoming_labelled_batch.csv")
    validation = validate_batch(incoming)
    print("TASK A - Incoming batch validation:", json.dumps(validation, indent=2))

    champion_path = MODELS / "housing_champion.joblib"
    if not champion_path.exists():
        raise FileNotFoundError("Champion missing. Run `python train.py` first.")
    champion = joblib.load(champion_path)
    holdout = pd.read_csv(ARTIFACTS / "quality_holdout.csv")
    historical = pd.read_csv(ARTIFACTS / "historical_training_data.csv")
    new_batch = pd.read_csv(ARTIFACTS / "incoming_labelled_batch.csv")

    incoming_metrics = metrics_for(champion, new_batch)
    champion_metrics = metrics_for(champion, holdout)
    eval_report = {
        "incoming_batch_dataset": "incoming_labelled_batch.csv",
        "incoming_batch_rows": int(len(new_batch)),
        "champion_on_incoming_batch": incoming_metrics,
        "evaluation_dataset": "quality_holdout.csv",
        "holdout_rows": int(len(holdout)),
        "champion": champion_metrics
    }
    (ARTIFACTS / "evaluation_report.json").write_text(json.dumps(eval_report, indent=2))
    print("TASK B - Incoming batch evaluation:", json.dumps(incoming_metrics, indent=2))
    print("TASK B - Champion evaluation:", json.dumps(champion_metrics, indent=2))

    # C: combine historical training records with the incoming labelled batch.
    combined = pd.concat([historical, new_batch], ignore_index=True)
    candidate = make_candidate()
    candidate.fit(combined.drop(columns=[TARGET]), combined[TARGET])
    candidate_path = MODELS / "housing_candidate.joblib"
    joblib.dump(candidate, candidate_path)
    candidate_metrics = metrics_for(candidate, holdout)
    eval_report["candidate"] = candidate_metrics
    eval_report["training_rows_after_retraining"] = int(len(combined))
    (ARTIFACTS / "evaluation_report.json").write_text(json.dumps(eval_report, indent=2))
    print("TASK C - Candidate saved:", candidate_path)
    print("Candidate metrics:", json.dumps(candidate_metrics, indent=2))

    # D: promotion is a recorded decision, not an automatic champion overwrite.
    threshold = champion_metrics["rmse"] * (1 - MIN_RMSE_IMPROVEMENT)
    eligible = candidate_metrics["rmse"] <= threshold
    decision = {
        "decision": "ELIGIBLE_FOR_PROMOTION" if eligible else "RETAIN_CHAMPION",
        "champion_rmse": champion_metrics["rmse"],
        "candidate_rmse": candidate_metrics["rmse"],
        "required_max_candidate_rmse": threshold,
        "minimum_rmse_improvement": MIN_RMSE_IMPROVEMENT,
        "selected_metric": "RMSE",
        "reason": ("Candidate RMSE improves by at least 1%; eligible for explicit promotion."
                   if eligible else
                   "Candidate did not achieve the required 1% RMSE improvement; champion retained."),
        "champion_overwritten": False
    }
    (ARTIFACTS / "quality_gate_decision.json").write_text(json.dumps(decision, indent=2))
    print("TASK D - Quality gate:", json.dumps(decision, indent=2))

    try:
        if mlflow is None:
            raise RuntimeError('MLflow is not installed; install requirements.txt to enable tracking')
        mlflow.set_tracking_uri("file:" + str(ROOT / "mlruns"))
        mlflow.set_experiment("housing-price-monitoring")
        with mlflow.start_run(run_name="monitor-and-retrain") as run:
            mlflow.log_params({"candidate_model": "RandomForestRegressor",
                               "candidate_n_estimators": 350,
                               "training_rows": len(combined),
                               "min_rmse_improvement": MIN_RMSE_IMPROVEMENT})
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})
            mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_metrics.items()})
            mlflow.log_metric("candidate_eligible", int(eligible))
            mlflow.log_artifact(str(candidate_path))
            mlflow.log_artifact(str(ARTIFACTS / "evaluation_report.json"))
            mlflow.log_artifact(str(ARTIFACTS / "quality_gate_decision.json"))
            decision["mlflow_run_id"] = run.info.run_id
            (ARTIFACTS / "quality_gate_decision.json").write_text(json.dumps(decision, indent=2))
    except Exception as exc:
        print(f"MLflow tracking unavailable; local reports still saved: {exc}")

    print("\nWORKFLOW COMPLETE. Champion remains unchanged.")
    print("Reports: artifacts/evaluation_report.json, artifacts/quality_gate_decision.json")

if __name__ == "__main__":
    main()
