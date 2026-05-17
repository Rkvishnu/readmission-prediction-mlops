from typing import Any

from prefect import flow, task

from readmission.models.train import run_training
from scripts.validate_training_outputs import (
    DEFAULT_COMPARISON_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    validate_outputs,
)


@task
def train_task() -> dict[str, Any]:
    return run_training()


@task
def validate_training_task(min_roc_auc: float = 0.65, min_recall: float = 0.55):
    return validate_outputs(
        metrics_path=DEFAULT_METRICS_PATH,
        comparison_path=DEFAULT_COMPARISON_PATH,
        model_path=DEFAULT_MODEL_PATH,
        min_roc_auc=min_roc_auc,
        min_recall=min_recall,
    )


@flow(name="readmission-training")
def training_flow(min_roc_auc: float = 0.65, min_recall: float = 0.55) -> dict[str, Any]:
    train_result = train_task()
    validation_result = validate_training_task(min_roc_auc=min_roc_auc, min_recall=min_recall)
    return {"training": train_result, "validation": validation_result}


if __name__ == "__main__":
    training_flow()
