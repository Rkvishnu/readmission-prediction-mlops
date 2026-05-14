import joblib
import pandas as pd

from readmission.data.preprocess import apply_feature_types
from readmission.utils.config import load_yaml
from readmission.utils.paths import CONFIG_DIR


def load_model(path: str):
    return joblib.load(path)


def predict_readmission(model, records: list[dict]) -> list[float]:
    data = pd.DataFrame(records)
    data_config = load_yaml(CONFIG_DIR / "data_config.yaml")
    data = apply_feature_types(data, data_config.get("categorical_columns", []))
    return model.predict_proba(data)[:, 1].tolist()
