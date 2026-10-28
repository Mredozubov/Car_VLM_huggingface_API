from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from backend.app.core.config import MAX_UPLOAD_BYTES
from backend.app.schemas.api import (
    DetectionResponse,
    DetectionListResponse,
    FeedbackResponse,
    HealthResponse,
    PredictionResponse,
    ReferenceResponse,
)
from backend.app.services.detector import VehicleNotDetectedError
from backend.app.services.inference import image_fingerprint
from backend.app.services.prediction_cache import CachedPrediction
from backend.app.utils.images import ImageValidationError, load_validated_image_bytes

router = APIRouter(prefix="/api/v1")


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"Image is too large. Maximum size is {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB."
        )
    return data


@router.post("/detect", response_model=DetectionListResponse)
async def detect_vehicles(
    request: Request,
    file: Annotated[UploadFile, File(description="One JPG, PNG, or WEBP vehicle image")],
) -> DetectionListResponse:
    try:
        data = await _read_upload(file)
        image = load_validated_image_bytes(
            data, filename=file.filename or "", content_type=file.content_type
        )
        detections = await run_in_threadpool(
            request.app.state.inference_service.detect_all, image
        )
    except ImageValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return DetectionListResponse(
        image_width=image.width,
        image_height=image.height,
        detections=[
            DetectionResponse(
                vehicle_type=item.label,
                confidence_percent=round(item.confidence * 100, 2),
                bounding_box=item.bounding_box,
            )
            for item in detections
        ],
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_vehicle(
    request: Request,
    file: Annotated[UploadFile, File(description="One JPG, PNG, or WEBP vehicle image")],
    selection_left: Annotated[int | None, Form(ge=0)] = None,
    selection_top: Annotated[int | None, Form(ge=0)] = None,
    selection_right: Annotated[int | None, Form(ge=1)] = None,
    selection_bottom: Annotated[int | None, Form(ge=1)] = None,
    selection_mask: Annotated[UploadFile | None, File()] = None,
) -> PredictionResponse:
    try:
        data = await _read_upload(file)
        image = load_validated_image_bytes(
            data, filename=file.filename or "", content_type=file.content_type
        )
        coordinates = (selection_left, selection_top, selection_right, selection_bottom)
        if any(value is not None for value in coordinates):
            if not all(value is not None for value in coordinates):
                raise ImageValidationError("All four selection coordinates are required.")
            mask = None
            if selection_mask is not None:
                mask_data = await _read_upload(selection_mask)
                mask = load_validated_image_bytes(
                    mask_data,
                    filename=selection_mask.filename or "selection.png",
                    content_type=selection_mask.content_type or "image/png",
                )
            box = tuple(int(value) for value in coordinates if value is not None)
            outcome = await run_in_threadpool(
                request.app.state.inference_service.predict_selection, image, box, mask
            )
        else:
            outcome = await run_in_threadpool(
                request.app.state.inference_service.predict, image
            )
    except ImageValidationError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except VehicleNotDetectedError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    prediction_id = f"pred_{uuid.uuid4().hex}"
    request.app.state.prediction_cache.put(
        prediction_id,
        CachedPrediction(
            classification=outcome.classification,
            image_fingerprint=image_fingerprint(data),
            created_at=time.time(),
            bounding_box=outcome.detection.bounding_box,
        ),
    )
    detection = outcome.detection
    return PredictionResponse(
        prediction_id=prediction_id,
        detection=DetectionResponse(
            vehicle_type=detection.label,
            confidence_percent=round(detection.confidence * 100, 2),
            bounding_box=detection.bounding_box,
        ),
        classification=outcome.classification.to_dict(),
        attention_image_data_url=outcome.attention_data_url,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: Request,
    prediction_id: Annotated[str, Form(min_length=6, max_length=128)],
    selected_rank: Annotated[int, Form(ge=1, le=5)],
    user_confidence: Annotated[str, Form(pattern="^(certain|likely|unsure)$")] = "certain",
    consent_to_training: Annotated[bool, Form()] = False,
    file: Annotated[UploadFile | None, File()] = None,
) -> FeedbackResponse:
    cached = request.app.state.prediction_cache.get(prediction_id)
    if cached is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction was not found or has expired. Run prediction again.",
        )
    if selected_rank > len(cached.classification.predictions):
        raise HTTPException(status_code=422, detail="Selected prediction rank is unavailable.")

    original_image = None
    crop_image = None
    if consent_to_training:
        if file is None:
            raise HTTPException(
                status_code=422,
                detail="The original image must be resubmitted when training consent is enabled.",
            )
        try:
            data = await _read_upload(file)
            if image_fingerprint(data) != cached.image_fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail="The contributed image does not match the prediction image.",
                )
            original_image = load_validated_image_bytes(
                data, filename=file.filename or "", content_type=file.content_type
            )
            if cached.bounding_box is not None:
                crop_image = original_image.crop(cached.bounding_box)
            else:
                crop_image = await run_in_threadpool(
                    request.app.state.inference_service.detect_crop, original_image
                )
        except ImageValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except VehicleNotDetectedError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    prediction = cached.classification.predictions[selected_rank - 1]
    confirmation = await run_in_threadpool(
        request.app.state.feedback_store.save_confirmation,
        prediction_id=prediction_id,
        selected_rank=selected_rank,
        prediction=prediction,
        user_confidence=user_confidence,
        consent_to_training=consent_to_training,
        original_image=original_image,
        crop_image=crop_image,
        model_id=cached.classification.model_id,
    )
    return FeedbackResponse(
        confirmation_id=confirmation.confirmation_id,
        prediction_id=prediction_id,
        selected_rank=selected_rank,
        image_stored=confirmation.image_stored,
    )


@router.get("/references/{reference_id}", response_model=ReferenceResponse)
async def get_reference(request: Request, reference_id: str) -> ReferenceResponse:
    reference = request.app.state.reference_gallery.get(reference_id)
    if reference is None:
        raise HTTPException(status_code=404, detail="Reference not found.")
    return ReferenceResponse(
        reference_id=reference.reference_id,
        make=reference.make,
        model=reference.model,
        generation=reference.generation,
        year_start=reference.year_start,
        year_end=reference.year_end,
        source_url=reference.source_url,
        license=reference.license,
        attribution=reference.attribution,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    service = request.app.state.inference_service
    return HealthResponse(
        status="ok",
        models_loaded=True,
        classifier_model_id=service.classifier.model_id,
        reference_count=len(request.app.state.reference_gallery.records),
        generation_count=len(request.app.state.generation_catalog.generations),
    )
