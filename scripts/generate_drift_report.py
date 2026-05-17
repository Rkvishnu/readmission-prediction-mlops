import argparse
from pathlib import Path

from readmission.monitoring.drift_report import generate_drift_report
from readmission.utils.config import load_yaml
from readmission.utils.paths import CONFIG_DIR, REPORTS_DIR


def parse_args() -> argparse.Namespace:
    data_config = load_yaml(CONFIG_DIR / "data_config.yaml")
    default_output = REPORTS_DIR / "drift" / "data_drift_report.html"
    default_summary = REPORTS_DIR / "drift" / "data_drift_summary.json"

    parser = argparse.ArgumentParser(description="Generate an Evidently data drift report.")
    parser.add_argument("--reference-path", default=data_config["reference_data_path"])
    parser.add_argument("--current-path", default=data_config["processed_test_path"])
    parser.add_argument("--output-path", default=str(default_output))
    parser.add_argument("--summary-path", default=str(default_summary))
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Optional monitored feature subset. Defaults to stable clinical/API features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = generate_drift_report(
        reference_path=args.reference_path,
        current_path=args.current_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        columns=args.columns,
    )
    summary = artifacts["result"]
    print(f"Drift status: {summary['status']}")
    print(f"Drifted columns: {summary['drifted_columns_count']}")
    print(f"HTML report: {Path(artifacts['html_report'])}")
    print(f"JSON summary: {Path(artifacts['summary'])}")


if __name__ == "__main__":
    main()
