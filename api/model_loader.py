import json
import os
from pathlib import Path

import mlflow

from readmission.models.predict import load_model
from readmission.models.registry import load_registered_model


MODEL_METRICS_PATH = Path("reports/metrics/training_metrics.json")


def get_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        return load_registered_model(model_uri)

    local_model_path = Path(os.getenv("LOCAL_MODEL_PATH", "models/readmission_model.joblib"))
    if not local_model_path.exists():
        return None
    return load_model(local_model_path)


def get_model_info() -> dict:
    local_model_path = Path(os.getenv("LOCAL_MODEL_PATH", "models/readmission_model.joblib"))
    model_uri = os.getenv("MODEL_URI")
    info = {
        "model_name": None,
        "status": "missing",
        "metrics": {},
        "model_artifact_available": bool(model_uri) or local_model_path.exists(),
        "model_source": model_uri or str(local_model_path),
    }

    if MODEL_METRICS_PATH.exists():
        metrics_payload = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
        info["model_name"] = metrics_payload.get("model_name")
        info["status"] = metrics_payload.get("status", "unknown")
        info["metrics"] = {
            key: value
            for key, value in metrics_payload.items()
            if isinstance(value, int | float) and key not in {"model_name", "status"}
        }
    elif local_model_path.exists():
        info["status"] = "available"

    return info
