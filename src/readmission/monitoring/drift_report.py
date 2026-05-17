import json
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


DEFAULT_MONITORED_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]


def _load_feature_frame(path: str | Path, columns: list[str]) -> pd.DataFrame:
    data = pd.read_csv(path)
    available_columns = [column for column in columns if column in data.columns]
    if not available_columns:
        raise ValueError(f"No monitored columns found in {path}")
    return data[available_columns]


def summarize_drift(snapshot: dict[str, Any]) -> dict[str, Any]:
    metrics = snapshot.get("metrics", [])
    drift_count = 0
    drift_share = 0.0
    drifted_columns = []

    for metric in metrics:
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value")

        if metric_name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            drift_count = int(value.get("count", 0))
            drift_share = float(value.get("share", 0.0))
            continue

        if config.get("type") == "evidently:metric_v2:ValueDrift":
            threshold = float(config.get("threshold", 0.05))
            method = config.get("method", "")
            if not isinstance(value, int | float):
                continue

            drift_detected = value < threshold if "p_value" in method else value > threshold
            if drift_detected:
                drifted_columns.append(
                    {
                        "column": config.get("column"),
                        "score": float(value),
                        "threshold": threshold,
                        "method": method,
                    }
                )

    return {
        "status": "drift_detected" if drift_count else "no_drift_detected",
        "drifted_columns_count": drift_count,
        "drifted_columns_share": drift_share,
        "drifted_columns": drifted_columns,
    }


def generate_drift_report(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    columns: list[str] | None = None,
) -> dict[str, Path | dict[str, Any]]:
    monitored_columns = columns or DEFAULT_MONITORED_FEATURES
    reference = _load_feature_frame(reference_path, monitored_columns)
    current = _load_feature_frame(current_path, monitored_columns)
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output))

    summary = summarize_drift(result.dict())
    summary.update(
        {
            "reference_path": str(reference_path),
            "current_path": str(current_path),
            "monitored_columns": list(reference.columns),
            "reference_rows": len(reference),
            "current_rows": len(current),
            "html_report_path": str(output),
        }
    )

    summary_output = Path(summary_path) if summary_path else output.with_suffix(".json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {"html_report": output, "summary": summary_output, "result": summary}
