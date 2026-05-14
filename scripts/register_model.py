import json

import mlflow

from readmission.models.registry import register_and_promote_best_model
from readmission.utils.config import load_yaml
from readmission.utils.paths import CONFIG_DIR


def main() -> None:
    model_config = load_yaml(CONFIG_DIR / "model_config.yaml")
    params = load_yaml("params.yaml")

    result = register_and_promote_best_model(
        experiment_name=model_config["experiment_name"],
        registered_model_name=model_config["registered_model_name"],
        primary_metric=params["evaluation"]["primary_metric"],
        alias="production",
    )
    print(json.dumps(result, indent=2))
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
