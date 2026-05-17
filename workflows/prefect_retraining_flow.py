import os
from pathlib import Path
from typing import Any

from prefect import flow, task

from readmission.data.preprocess import main as preprocess_data
from scripts.register_model import main as register_model
from workflows.prefect_monitoring_flow import monitoring_flow
from workflows.prefect_training_flow import training_flow


@task
def prepare_data_task() -> str:
    preprocess_data()
    return "prepared"


@task
def should_retrain_task(drift_summary: dict[str, Any], force_retrain: bool = False) -> bool:
    return force_retrain or drift_summary.get("status") == "drift_detected"


@task
def register_model_task() -> str:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return "skipped: MLFLOW_TRACKING_URI is not set"
    register_model()
    return "registered"


@flow(name="readmission-retraining")
def retraining_flow(
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    force_retrain: bool = False,
    register_on_success: bool = False,
    min_roc_auc: float = 0.65,
    min_recall: float = 0.55,
) -> dict[str, Any]:
    prepare_data_task()
    drift_summary = monitoring_flow(
        reference_path=reference_path,
        current_path=current_path,
        output_path=output_path,
        summary_path=summary_path,
    )
    should_retrain = should_retrain_task(drift_summary, force_retrain=force_retrain)

    result: dict[str, Any] = {
        "drift": drift_summary,
        "retrained": should_retrain,
        "training": None,
        "registration": "skipped",
    }
    if should_retrain:
        result["training"] = training_flow(min_roc_auc=min_roc_auc, min_recall=min_recall)
        if register_on_success:
            result["registration"] = register_model_task()

    return result


if __name__ == "__main__":
    retraining_flow()
