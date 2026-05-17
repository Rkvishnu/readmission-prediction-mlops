import mlflow


def load_registered_model(model_uri: str):
    if model_uri.startswith("models:/") and "@" in model_uri:
        model_reference = model_uri.removeprefix("models:/")
        model_name, alias = model_reference.split("@", maxsplit=1)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(model_name, alias)
        return mlflow.sklearn.load_model(f"runs:/{model_version.run_id}/model")

    return mlflow.sklearn.load_model(model_uri)


def get_best_run(
    experiment_name: str,
    primary_metric: str,
) -> tuple[str, dict[str, float], dict[str, str]]:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{primary_metric} DESC", "attributes.start_time DESC"],
        max_results=50,
    )
    runs = [run for run in runs if primary_metric in run.data.metrics]
    if not runs:
        raise ValueError(
            f"No runs with metric '{primary_metric}' found in experiment '{experiment_name}'"
        )

    best_run = runs[0]
    return best_run.info.run_id, best_run.data.metrics, best_run.data.params


def register_and_promote_best_model(
    experiment_name: str,
    registered_model_name: str,
    primary_metric: str = "roc_auc",
    alias: str = "production",
) -> dict:
    client = mlflow.tracking.MlflowClient()
    run_id, metrics, params = get_best_run(experiment_name, primary_metric)
    model_uri = f"runs:/{run_id}/model"

    try:
        client.create_registered_model(registered_model_name)
    except mlflow.exceptions.MlflowException:
        pass

    model_version = client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=run_id,
    )
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=model_version.version,
    )

    return {
        "registered_model_name": registered_model_name,
        "alias": alias,
        "version": model_version.version,
        "run_id": run_id,
        "model_uri": f"models:/{registered_model_name}@{alias}",
        "metrics": metrics,
        "params": params,
    }
