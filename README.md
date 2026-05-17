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

## CI/CD

GitHub Actions runs the project quality gate on every push and pull request:

- installs the Python project with development dependencies
- runs Ruff linting across `src`, `api`, `tests`, `workflows`, and `scripts`
- runs the unit and API test suite
- builds the FastAPI Docker image from `deployment/docker/Dockerfile.api`
- publishes the API image to GitHub Container Registry on default-branch pushes

The workflow lives at `.github/workflows/ci.yml`. Pull requests build the image without pushing it. Pushes to the default branch publish tags such as `latest`, branch name, and `sha-<commit>` to GitHub Container Registry.

Additional lifecycle workflows:

- `.github/workflows/train.yaml`: manually runs data preparation, model training, and metric threshold validation.
- `.github/workflows/monitor.yml`: manually or weekly generates an Evidently drift report and uploads it as a workflow artifact.

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

## Kubernetes

You can deploy the API and supporting services to a local Kubernetes cluster using Docker Desktop Kubernetes or another local cluster.

If `kubectl` returns `connection refused`, enable Kubernetes in Docker Desktop under Settings > Kubernetes, wait for it to start, and confirm:

```bash
kubectl config use-context docker-desktop
kubectl cluster-info
```

Build the local API image:

```bash
make k8s-build
```

Deploy the raw Kubernetes manifests:

```bash
make k8s-deploy
make k8s-status
```

Wait until the `readmission-api` pods are ready, then forward the service:

```bash
make k8s-port-forward
```

Open the API at `http://localhost:8000/docs` and test:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

The Kubernetes API deployment includes readiness and liveness probes, CPU and memory requests/limits, and Prometheus scrape annotations for `/metrics/`.

To deploy the Helm chart instead:

```bash
make helm-template
make helm-deploy
```

Clean up the raw manifest deployment:

```bash
make k8s-delete
```

## Drift Monitoring

Generate an Evidently data drift report by comparing the reference training sample with the current processed test data:

```bash
make drift
```

Generated artifacts:

- `reports/drift/data_drift_report.html`
- `reports/drift/data_drift_summary.json`

The report monitors stable clinical and serving features such as hospital stay length, lab procedure count, medication count, visit history, diagnosis count, and admission/discharge/source IDs. The JSON summary is designed for automation, while the HTML report is useful for review and portfolio screenshots.

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
