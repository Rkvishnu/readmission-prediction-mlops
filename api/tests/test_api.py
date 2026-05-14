from fastapi.testclient import TestClient

import api.model_loader
import api.routes.predict
from api.main import app


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint():
    client = TestClient(app)
    response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "metrics" in payload
    assert "model_artifact_available" in payload


def test_predict_endpoint_returns_risk_categories(monkeypatch):
    class FakeModel:
        pass

    monkeypatch.setattr(api.routes.predict, "get_model", lambda: FakeModel())
    monkeypatch.setattr(api.routes.predict, "predict_readmission", lambda model, records: [0.2, 0.5, 0.8])

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "records": [
                {"age": "[50-60)", "num_medications": 10},
                {"age": "[60-70)", "num_medications": 12},
                {"age": "[70-80)", "num_medications": 20},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [
            {"risk_score": 0.2, "risk_category": "low"},
            {"risk_score": 0.5, "risk_category": "medium"},
            {"risk_score": 0.8, "risk_category": "high"},
        ]
    }


def test_predict_endpoint_returns_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(api.routes.predict, "get_model", lambda: None)

    client = TestClient(app)
    response = client.post("/predict", json={"records": [{"age": "[50-60)"}]})

    assert response.status_code == 503
