from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    records: list[dict] = Field(min_length=1)


class PredictionResult(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_category: str


class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]


class ModelInfoResponse(BaseModel):
    model_name: str | None = None
    status: str
    metrics: dict[str, float] = Field(default_factory=dict)
    model_artifact_available: bool
    model_source: str
