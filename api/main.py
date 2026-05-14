from fastapi import FastAPI
from prometheus_client import make_asgi_app

from api.routes.health import router as health_router
from api.routes.model_info import router as model_info_router
from api.routes.predict import router as predict_router

app = FastAPI(title="Hospital Readmission Risk API", version="0.1.0")
app.include_router(health_router)
app.include_router(model_info_router)
app.include_router(predict_router)
app.mount("/metrics", make_asgi_app())
