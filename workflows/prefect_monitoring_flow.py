from pathlib import Path
from typing import Any

from prefect import flow, task

from readmission.monitoring.drift_report import generate_drift_report
from readmission.utils.config import load_yaml
from readmission.utils.paths import CONFIG_DIR, REPORTS_DIR


@task
def generate_drift_report_task(
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    data_config = load_yaml(CONFIG_DIR / "data_config.yaml")
    artifacts = generate_drift_report(
        reference_path=reference_path or data_config["reference_data_path"],
        current_path=current_path or data_config["processed_test_path"],
        output_path=output_path or REPORTS_DIR / "drift" / "data_drift_report.html",
        summary_path=summary_path or REPORTS_DIR / "drift" / "data_drift_summary.json",
    )
    return artifacts["result"]


@flow(name="readmission-monitoring")
def monitoring_flow(
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    return generate_drift_report_task(
        reference_path=reference_path,
        current_path=current_path,
        output_path=output_path,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    monitoring_flow()
