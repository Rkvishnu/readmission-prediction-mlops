from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from readmission.data.ingest import load_raw_data
from readmission.data.validate import validate_columns
from readmission.utils.config import load_yaml
from readmission.utils.paths import CONFIG_DIR


def normalize_target(value: object) -> int:
    return 1 if str(value).strip() == "<30" else 0


def preprocess_dataframe(data: pd.DataFrame, target_column: str = "readmitted") -> pd.DataFrame:
    validate_columns(data, {target_column})
    cleaned = data.copy()
    cleaned = cleaned.replace("?", pd.NA)
    cleaned[target_column] = cleaned[target_column].map(normalize_target)
    return cleaned


def apply_feature_types(data: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    typed = data.copy()
    for column in categorical_columns:
        if column in typed.columns:
            typed[column] = typed[column].astype("string")
    return typed


def prepare_modeling_data(
    data: pd.DataFrame,
    target_column: str,
    drop_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    modeled = preprocess_dataframe(data, target_column)
    modeled = modeled.drop(columns=[column for column in drop_columns if column in modeled.columns])
    if "gender" in modeled.columns:
        modeled = modeled[modeled["gender"] != "Unknown/Invalid"]

    return apply_feature_types(modeled, categorical_columns).reset_index(drop=True)


def create_train_test_split(data: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[target_column],
    )
    return train, test


def main() -> None:
    data_config = load_yaml(CONFIG_DIR / "data_config.yaml")
    params = load_yaml("params.yaml")
    raw_path = Path(data_config["raw_data_path"])
    train_path = Path(data_config["processed_train_path"])
    test_path = Path(data_config["processed_test_path"])

    data = prepare_modeling_data(
        load_raw_data(raw_path),
        data_config["target_column"],
        data_config.get("drop_columns", []),
        data_config.get("categorical_columns", []),
    )
    train, test = create_train_test_split(
        data,
        data_config["target_column"],
        params["data"]["test_size"],
        params["data"]["random_state"],
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    Path(data_config["reference_data_path"]).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    train.drop(columns=[data_config["target_column"]]).sample(
        n=min(5000, len(train)),
        random_state=params["data"]["random_state"],
    ).to_csv(data_config["reference_data_path"], index=False)


if __name__ == "__main__":
    main()
