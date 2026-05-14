from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def generate_drift_report(reference_path: str, current_path: str, output_path: str) -> Path:
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(output)
    return output

