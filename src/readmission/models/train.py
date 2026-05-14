import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from readmission.data.preprocess import apply_feature_types
from readmission.features.build_features import build_preprocessor
from readmission.utils.config import load_yaml
from readmission.utils.paths import CONFIG_DIR, MODEL_DIR, REPORTS_DIR


def build_classifier(model_name: str, model_params: dict):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            random_state=model_params["random_state"],
            class_weight="balanced",
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError(
                "XGBoost is installed but could not be imported. On macOS, install libomp "
                "with `brew install libomp`, then rerun training."
            ) from exc

        return XGBClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["learning_rate"],
            subsample=model_params["subsample"],
            colsample_bytree=model_params["colsample_bytree"],
            eval_metric=model_params["eval_metric"],
            random_state=model_params["random_state"],
            scale_pos_weight=model_params.get("scale_pos_weight", 1.0),
            n_jobs=1,
        )
    if model_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=model_params["max_iter"],
            max_leaf_nodes=model_params["max_leaf_nodes"],
            learning_rate=model_params["learning_rate"],
            random_state=model_params["random_state"],
            class_weight="balanced",
        )
    raise ValueError(f"Unsupported model type: {model_name}")


def train_model(
    train_data: pd.DataFrame,
    target_column: str,
    params: dict,
    model_name: str = "random_forest",
) -> Pipeline:
    x_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    preprocessor = build_preprocessor(x_train, sparse_output=model_name != "gradient_boosting")
    classifier = build_classifier(model_name, params["models"][model_name])
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline.fit(x_train, y_train)


def evaluate_model(model: Pipeline, test_data: pd.DataFrame, target_column: str) -> dict[str, float]:
    x_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def save_evaluation_artifacts(
    model: Pipeline,
    test_data: pd.DataFrame,
    target_column: str,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    classification_report_path = output_dir / "classification_report.json"
    classification_report_path.write_text(
        json.dumps(classification_report(y_test, predictions, output_dict=True), indent=2),
        encoding="utf-8",
    )

    confusion_matrix_path = output_dir / "confusion_matrix.png"
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions),
        display_labels=["not_readmitted_30d", "readmitted_30d"],
    ).plot(values_format="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    roc_curve_path = output_dir / "roc_curve.png"
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_curve_path)
    plt.close()

    return {
        "classification_report": classification_report_path,
        "confusion_matrix": confusion_matrix_path,
        "roc_curve": roc_curve_path,
    }


def run_training() -> dict:
    data_config = load_yaml(CONFIG_DIR / "data_config.yaml")
    model_config = load_yaml(CONFIG_DIR / "model_config.yaml")
    params = load_yaml("params.yaml")

    categorical_columns = data_config.get("categorical_columns", [])
    train_data = apply_feature_types(pd.read_csv(data_config["processed_train_path"]), categorical_columns)
    test_data = apply_feature_types(pd.read_csv(data_config["processed_test_path"]), categorical_columns)
    target_column = data_config["target_column"]
    primary_metric = params["evaluation"]["primary_metric"]
    candidate_models = model_config["candidate_models"]

    mlflow.set_experiment(model_config["experiment_name"])
    model_results = []
    best_model = None
    best_result = None

    positive_count = train_data[target_column].sum()
    negative_count = len(train_data) - positive_count
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    for model_name in candidate_models:
        model_params = params["models"][model_name].copy()
        if model_name == "xgboost":
            model_params["scale_pos_weight"] = scale_pos_weight
        candidate_params = {**params, "models": {**params["models"], model_name: model_params}}

        with mlflow.start_run(run_name=model_name):
            try:
                model = train_model(train_data, target_column, candidate_params, model_name=model_name)
            except RuntimeError as exc:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("status", "skipped")
                mlflow.log_param("skip_reason", str(exc))
                model_results.append({"model_name": model_name, "status": "skipped", "reason": str(exc)})
                continue

            metrics = evaluate_model(model, test_data, target_column)
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

        result = {"model_name": model_name, "status": "trained", **metrics}
        model_results.append(result)
        if best_result is None or metrics[primary_metric] > best_result[primary_metric]:
            best_model = model
            best_result = result

    if best_model is None or best_result is None:
        raise RuntimeError("No candidate models were trained")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = REPORTS_DIR / "metrics"
    figures_dir = REPORTS_DIR / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = save_evaluation_artifacts(best_model, test_data, target_column, figures_dir)
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_param("model_name", best_result["model_name"])
        mlflow.log_metrics(
            {
                key: value
                for key, value in best_result.items()
                if key not in {"model_name", "status"}
            }
        )
        for artifact_path in artifact_paths.values():
            mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(best_model, artifact_path="model")

    joblib.dump(best_model, MODEL_DIR / "readmission_model.joblib")
    Path(metrics_dir / "training_metrics.json").write_text(
        json.dumps(best_result, indent=2),
        encoding="utf-8",
    )
    Path(metrics_dir / "model_comparison.json").write_text(
        json.dumps(model_results, indent=2),
        encoding="utf-8",
    )
    return {"best_model": best_result, "model_comparison": model_results}


if __name__ == "__main__":
    run_training()
