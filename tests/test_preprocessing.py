import pandas as pd

from readmission.data.preprocess import apply_feature_types, prepare_modeling_data, preprocess_dataframe


def test_preprocess_maps_readmission_target():
    data = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
    processed = preprocess_dataframe(data)
    assert processed["readmitted"].tolist() == [1, 0, 0]


def test_prepare_modeling_data_drops_ids_and_casts_categorical_ids():
    data = pd.DataFrame(
        {
            "encounter_id": [1],
            "patient_nbr": [2],
            "admission_type_id": [1],
            "gender": ["Female"],
            "readmitted": ["<30"],
        }
    )
    processed = prepare_modeling_data(
        data,
        target_column="readmitted",
        drop_columns=["encounter_id", "patient_nbr"],
        categorical_columns=["admission_type_id"],
    )

    assert "encounter_id" not in processed.columns
    assert "patient_nbr" not in processed.columns
    assert str(processed["admission_type_id"].dtype) == "string"


def test_apply_feature_types_casts_configured_columns_only():
    data = pd.DataFrame({"admission_type_id": [1], "time_in_hospital": [3]})
    typed = apply_feature_types(data, ["admission_type_id"])

    assert str(typed["admission_type_id"].dtype) == "string"
    assert typed["time_in_hospital"].dtype == "int64"
