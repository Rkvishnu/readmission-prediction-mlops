# Hospital Readmission Risk Prediction MLOps

Production-style MLOps project for predicting whether a patient is likely to be readmitted within 30 days after hospital discharge.

## Objective

Build an end-to-end machine learning system that covers:

- data ingestion and validation
- preprocessing and feature engineering
- model training and evaluation
- experiment tracking with MLflow
- model serving with FastAPI
- containerization with Docker
- deployment with Kubernetes and Helm
- monitoring with Evidently, Prometheus, and Grafana
- retraining workflows with Prefect

## Tech Stack

- Python, pandas, scikit-learn, XGBoost
- MLflow for experiment tracking and model registry
- DVC for data and pipeline versioning
- FastAPI for model serving
- Docker and Docker Compose for local runtime
- Kubernetes and Helm for deployment
- GitHub Actions for CI/CD
- Evidently for drift reports
- Prometheus and Grafana for service metrics

## Project Layout

```text
configs/                 YAML configuration files
data/                    raw, interim, processed, and reference datasets
src/readmission/          reusable Python package
api/                     FastAPI inference service
workflows/               Prefect training, monitoring, and retraining flows
deployment/              Docker, Kubernetes, and Helm deployment assets
monitoring/              Prometheus, Grafana, and Evidently assets
reports/                 generated metrics, plots, and drift reports
tests/                   unit and integration tests
scripts/                 developer convenience scripts
```

## Quickstart

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

Download the dataset:

```bash
python scripts/download_data.py
```

Prepare the processed data:

```bash
python -m readmission.data.preprocess
```

Train the first model:

```bash
make train
```

Or run the reproducible DVC pipeline:

```bash
dvc repro
```

The training pipeline compares configured candidate models and saves the best model by ROC AUC to `models/readmission_model.joblib`. Current generated evaluation outputs:

- `reports/metrics/training_metrics.json`
- `reports/metrics/model_comparison.json`
- `reports/figures/classification_report.json`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`

Note: XGBoost requires OpenMP on macOS. If the XGBoost run is skipped with a `libomp` error, install it with `brew install libomp` and rerun `dvc repro`.

Register and promote the best MLflow run:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/register_model.py
```

The script registers the best run as `readmission-risk` and assigns the `production` alias. The API can load this model with:

```bash
export MODEL_URI=models:/readmission-risk@production
```

Run checks:

```bash
make lint
make test
```

Run the API locally:

```bash
make api
```

The initial API exposes:

- `GET /health`
- `POST /predict`
- `GET /model-info`
- `GET /metrics`

Example prediction response:

```json
{
  "predictions": [
    {
      "risk_score": 0.4306,
      "risk_category": "medium"
    }
  ]
}
```

Risk categories are derived from the predicted 30-day readmission probability:

- `low`: less than `0.30`
- `medium`: `0.30` to less than `0.60`
- `high`: `0.60` and above

## Docker Compose

After training a model with `dvc repro`, run the local MLOps stack:

```bash
docker compose up --build
```

Services:

- FastAPI: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Grafana login:

- username: `admin`
- password: `admin`

Stop the stack:

```bash
docker compose down
```

## Dataset

This project is designed around the Diabetes 130-US hospitals readmission dataset. Raw datasets are intentionally not committed to git. Place source files under `data/raw/`, then run the preprocessing and training pipeline.

Source: UCI Machine Learning Repository, Diabetes 130-US Hospitals for Years 1999-2008.

## Development Roadmap

1. Project scaffolding and environment setup
2. Data ingestion and validation
3. Exploratory analysis
4. Preprocessing and feature engineering
5. Baseline model training
6. MLflow experiment tracking
7. DVC pipeline
8. FastAPI model serving
9. Dockerization
10. CI with GitHub Actions
11. Kubernetes deployment
12. Monitoring and drift reports
13. Scheduled retraining workflow
