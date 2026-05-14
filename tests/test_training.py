import pandas as pd

from readmission.models.train import build_classifier, train_model


def test_train_model_returns_fitted_pipeline():
    data = pd.DataFrame(
        {
            "age": ["[50-60)", "[60-70)", "[70-80)", "[80-90)"],
            "num_medications": [10, 12, 8, 20],
            "readmitted": [0, 1, 0, 1],
        }
    )
    params = {"models": {"random_forest": {"n_estimators": 5, "max_depth": 2, "random_state": 42}}}
    model = train_model(data, "readmitted", params)
    assert hasattr(model, "predict")


def test_build_classifier_supports_xgboost():
    try:
        classifier = build_classifier(
            "xgboost",
            {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "eval_metric": "logloss",
                "random_state": 42,
            },
        )
    except RuntimeError as exc:
        assert "libomp" in str(exc)
    else:
        assert classifier.__class__.__name__ == "XGBClassifier"


def test_build_classifier_supports_gradient_boosting():
    classifier = build_classifier(
        "gradient_boosting",
        {
            "max_iter": 5,
            "max_leaf_nodes": 15,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    )
    assert classifier.__class__.__name__ == "HistGradientBoostingClassifier"
