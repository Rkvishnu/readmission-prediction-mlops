from fastapi import APIRouter

from api.model_loader import get_model_info
from api.schemas import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**get_model_info())
