import pandas as pd

from readmission.models.predict import predict_readmission
from readmission.models.train import train_model


def test_predict_readmission_returns_probability():
    data = pd.DataFrame(
        {
            "age": ["[50-60)", "[60-70)", "[70-80)", "[80-90)"],
            "num_medications": [10, 12, 8, 20],
            "readmitted": [0, 1, 0, 1],
        }
    )
    params = {"models": {"random_forest": {"n_estimators": 5, "max_depth": 2, "random_state": 42}}}
    model = train_model(data, "readmitted", params)
    risks = predict_readmission(model, [{"age": "[50-60)", "num_medications": 10}])
    assert len(risks) == 1
    assert 0 <= risks[0] <= 1
