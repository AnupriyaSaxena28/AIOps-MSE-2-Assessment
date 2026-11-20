# Housing Price MLOps Monitoring and Retraining

## Assessment summary

This project originally served an Iris classification model. It has been adapted to
the supplied housing dataset, which is a labelled **regression** problem:

- Dataset: `data/Housing-selected-columns.csv`
- Rows: 545
- Target: `price`
- Features: 9 total: 4 numeric and 5 categorical
- Model: scikit-learn `RandomForestRegressor`
- Serving API: FastAPI
- Tracking: local MLflow when installed and available

All five assessment tasks are implemented:

| Task | Implementation | Evidence |
| --- | --- | --- |
| A. Prepare new data | Validates required columns, missing values, duplicate rows, positive prices, row/column counts, and five-bin price distribution. | `monitor.py`, `artifacts/incoming_labelled_batch.csv` |
| B. Evaluate existing model | Loads the champion and reports MAE, RMSE, and R² on the incoming labelled batch. It also evaluates the champion on an untouched holdout for a fair gate comparison. | `artifacts/evaluation_report.json` |
| C. Retrain candidate | Combines historical training data with the incoming labelled batch, fits the same preprocessing pipeline, and saves a separate candidate model. | `models/housing_candidate.joblib` |
| D. Quality gate | Requires at least 1% RMSE improvement. Records eligibility and never overwrites the champion automatically. | `artifacts/quality_gate_decision.json` |
| E. Demonstration | Provides reproducible commands, saved reports, model artifacts, API endpoints, and tests. | This README, `artifacts/`, and `test/` |

## Reproducible data split

Because no separate historical housing file was provided, `train.py` creates a
deterministic simulation using random seed `42`:

1. 70% historical training data: 381 rows
2. 15% incoming labelled batch: 82 rows
3. 15% untouched quality holdout: 82 rows

The holdout is not used to fit either model. Both champion and candidate are
evaluated on it using the same rows, preventing an unfair comparison.

## Task A — Prepare and validate new data

`monitor.py` loads `artifacts/incoming_labelled_batch.csv` and checks:

- Required columns: `price`, `area`, `bedrooms`, `bathrooms`, `stories`,
  `mainroad`, `guestroom`, `basement`, `hotwaterheating`, and `airconditioning`
- Missing values
- Duplicate records
- Positive target prices
- Row count and column count
- Distribution of `price` across five bins

The dataset has no ID column, so duplicate complete records are checked instead of
duplicate IDs.

## Task B — Evaluate the existing champion

The existing model is loaded from `models/housing_champion.joblib`. The report
contains:

- MAE — average absolute prediction error
- RMSE — error in price units with greater penalty for large errors
- R² — explained variance

The champion is evaluated directly on the new labelled batch. It is also evaluated
on `quality_holdout.csv`; the holdout metrics are used for the champion/candidate
quality-gate comparison.

## Task C — Retrain a candidate model

The candidate training set is:

```text
historical_training_data.csv + incoming_labelled_batch.csv
381 + 82 = 463 rows
```

The candidate uses the same feature handling as the champion:

- Numeric features passed through unchanged
- Categorical features encoded with `OneHotEncoder(handle_unknown="ignore")`
- Random forest regression model

The candidate is saved separately as
`models/housing_candidate.joblib`. The champion remains untouched.

MLflow tracking is attempted for the initial training and monitoring/retraining
runs. If MLflow is unavailable, the scripts print the reason and still generate
the required local JSON evidence.

## Task D — Quality gate

RMSE is the primary quality metric because `price` is continuous, RMSE is expressed
in price units, and it penalizes large errors. The candidate must satisfy:

```text
candidate RMSE <= champion RMSE * 0.99
```

The current recorded execution produced:

| Metric | Champion | Candidate |
| --- | ---: | ---: |
| MAE | 1,013,349.54 | 1,022,696.28 |
| RMSE | 1,422,871.96 | 1,459,360.45 |
| R² | 0.5284 | 0.5039 |

The required maximum candidate RMSE was `1,408,643.24`. Since the candidate did
not improve RMSE by at least 1%, the recorded decision is:

```text
RETAIN_CHAMPION
```

This is a successful quality-gate outcome: the candidate was evaluated, rejected
for promotion based on the rule, and the champion was not overwritten.

## Task E — Run and demonstrate the project

### Windows PowerShell setup

Use Python 3.8+ where possible:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the complete monitoring workflow

```powershell
python train.py
python monitor.py
```

Expected outputs include:

- `models/housing_champion.joblib`
- `models/housing_candidate.joblib`
- `artifacts/historical_training_data.csv`
- `artifacts/incoming_labelled_batch.csv`
- `artifacts/quality_holdout.csv`
- `artifacts/initial_training_report.json`
- `artifacts/evaluation_report.json`
- `artifacts/quality_gate_decision.json`
- Optional local MLflow data under `mlruns/`

### Run the tests

```powershell
pytest -q
```

The API test suite covers health readiness, a valid prediction, and invalid input
validation.

### Run the API

After training:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

Example request to `POST /predict`:

```json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes"
}
```

Available endpoints:

- `GET /` — readiness response
- `GET /health` — API and champion model status
- `POST /predict` — returns `predicted_price`

## File guide

| File | Purpose |
| --- | --- |
| `train.py` | Creates the deterministic data split and trains the initial champion. |
| `monitor.py` | Performs Tasks A–D: validation, evaluation, retraining, and quality gate. |
| `main.py` | FastAPI application serving the champion model. |
| `data/Housing-selected-columns.csv` | Source labelled housing dataset. |
| `models/housing_champion.joblib` | Current serving model. |
| `models/housing_candidate.joblib` | Separately saved retrained candidate. |
| `artifacts/evaluation_report.json` | Incoming-batch and holdout evaluation metrics. |
| `artifacts/quality_gate_decision.json` | Promotion decision and threshold evidence. |
| `test/test_api.py` | API behavior tests. |

## Conclusion

The original Iris classifier has been consistently converted to housing-price
regression. The project validates new labelled data, evaluates the deployed
champion, retrains a separate candidate, applies a measurable RMSE quality gate,
records the decision, and preserves the existing champion when the candidate does
not meet the required improvement.
