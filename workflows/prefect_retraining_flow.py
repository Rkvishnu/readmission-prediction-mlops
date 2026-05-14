from prefect import flow

from workflows.prefect_monitoring_flow import monitoring_flow
from workflows.prefect_training_flow import training_flow


@flow(name="readmission-retraining")
def retraining_flow(reference_path: str, current_path: str, output_path: str):
    monitoring_flow(reference_path, current_path, output_path)
    return training_flow()

