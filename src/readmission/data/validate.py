import pandas as pd


REQUIRED_COLUMNS = {"readmitted"}


def validate_columns(data: pd.DataFrame, required_columns: set[str] | None = None) -> None:
    required = required_columns or REQUIRED_COLUMNS
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

