from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.predict import router
from backend.app.core.config import ALLOWED_ORIGINS
from backend.app.services.feedback import FeedbackStore
from backend.app.services.inference import InferenceService
from backend.app.services.prediction_cache import PredictionCache


def create_app(
    *,
    inference_service: InferenceService | None = None,
    feedback_store: FeedbackStore | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = inference_service or await run_in_threadpool(InferenceService)
        app.state.inference_service = service
        app.state.feedback_store = feedback_store or FeedbackStore()
        app.state.prediction_cache = PredictionCache()
        app.state.reference_gallery = service.classifier.reference_gallery
        app.state.generation_catalog = service.classifier.generation_catalog
        yield

    application = FastAPI(
        title="AI Car Identifier API",
        version="0.3.0",
        description=(
            "Detect a vehicle and return five uncertain, evidence-backed identity candidates."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(router)
    return application


app = create_app()
