import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS_PATH = Path("reports/metrics/training_metrics.json")
DEFAULT_COMPARISON_PATH = Path("reports/metrics/model_comparison.json")
DEFAULT_MODEL_PATH = Path("models/readmission_model.joblib")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return json.loads(path.read_text())


def require_metric(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Metric '{metric_name}' is missing from {DEFAULT_METRICS_PATH}")

    value = metrics[metric_name]
    if not isinstance(value, int | float):
        raise TypeError(f"Metric '{metric_name}' must be numeric, got {type(value).__name__}")

    return float(value)


def validate_outputs(
    metrics_path: Path,
    comparison_path: Path,
    model_path: Path,
    min_roc_auc: float,
    min_recall: float,
) -> dict[str, float | str]:
    metrics = load_json(metrics_path)
    comparison = load_json(comparison_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Required model artifact is missing: {model_path}")

    if not isinstance(metrics, dict):
        raise TypeError(f"Expected metrics JSON object in {metrics_path}")

    if not isinstance(comparison, list) or not comparison:
        raise ValueError(f"Expected non-empty model comparison list in {comparison_path}")

    if metrics.get("status") != "trained":
        raise ValueError(f"Best model status must be 'trained', got {metrics.get('status')!r}")

    roc_auc = require_metric(metrics, "roc_auc")
    recall = require_metric(metrics, "recall")

    if roc_auc < min_roc_auc:
        raise ValueError(f"ROC AUC gate failed: {roc_auc:.4f} < {min_roc_auc:.4f}")

    if recall < min_recall:
        raise ValueError(f"Recall gate failed: {recall:.4f} < {min_recall:.4f}")

    return {
        "model_name": str(metrics.get("model_name", "unknown")),
        "roc_auc": roc_auc,
        "recall": recall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate training artifacts and model metrics.")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--comparison-path", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--min-roc-auc", type=float, default=0.65)
    parser.add_argument("--min-recall", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_outputs(
        metrics_path=args.metrics_path,
        comparison_path=args.comparison_path,
        model_path=args.model_path,
        min_roc_auc=args.min_roc_auc,
        min_recall=args.min_recall,
    )
    print(
        "Training validation passed: "
        f"model={result['model_name']}, "
        f"roc_auc={result['roc_auc']:.4f}, "
        f"recall={result['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
