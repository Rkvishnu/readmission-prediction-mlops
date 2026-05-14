import time

from fastapi import APIRouter, HTTPException

from api.model_loader import get_model
from api.schemas import PredictionRequest, PredictionResponse
from readmission.monitoring.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY
from readmission.models.predict import predict_readmission

router = APIRouter(tags=["prediction"])


def categorize_risk(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.6:
        return "medium"
    return "high"


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not available")

    start = time.perf_counter()
    risks = predict_readmission(model, request.records)
    PREDICTION_COUNTER.inc()
    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    return PredictionResponse(
        predictions=[
            {"risk_score": risk_score, "risk_category": categorize_risk(risk_score)}
            for risk_score in risks
        ]
    )
