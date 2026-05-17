import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from workflows.prefect_retraining_flow import retraining_flow  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Prefect drift-aware retraining flow.")
    parser.add_argument("--reference-path", type=Path, default=None)
    parser.add_argument("--current-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--register-on-success", action="store_true")
    parser.add_argument("--min-roc-auc", type=float, default=0.65)
    parser.add_argument("--min-recall", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = retraining_flow(
        reference_path=args.reference_path,
        current_path=args.current_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        force_retrain=args.force_retrain,
        register_on_success=args.register_on_success,
        min_roc_auc=args.min_roc_auc,
        min_recall=args.min_recall,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
