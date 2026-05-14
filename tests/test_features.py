import pandas as pd

from readmission.features.build_features import build_preprocessor


def test_build_preprocessor_fits_mixed_features():
    features = pd.DataFrame({"age": ["[50-60)", "[60-70)"], "num_medications": [10, 12]})
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)
    assert transformed.shape[0] == 2


def test_build_preprocessor_can_return_dense_features():
    features = pd.DataFrame({"age": ["[50-60)", "[60-70)"], "num_medications": [10, 12]})
    preprocessor = build_preprocessor(features, sparse_output=False)
    transformed = preprocessor.fit_transform(features)
    assert transformed.shape[0] == 2
    assert not hasattr(transformed, "toarray")
