from typing import Any

from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    vehicle_type: str
    confidence_percent: float
    bounding_box: tuple[int, int, int, int]


class DetectionListResponse(BaseModel):
    image_width: int
    image_height: int
    detections: list[DetectionResponse]


class PredictionResponse(BaseModel):
    prediction_id: str
    detection: DetectionResponse
    classification: dict[str, Any]
    attention_image_data_url: str


class FeedbackResponse(BaseModel):
    confirmation_id: str
    prediction_id: str
    selected_rank: int = Field(ge=1, le=5)
    image_stored: bool
    review_status: str = "pending"


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    classifier_model_id: str
    reference_count: int
    generation_count: int


class ReferenceResponse(BaseModel):
    reference_id: str
    make: str
    model: str
    generation: str
    year_start: int
    year_end: int
    source_url: str
    license: str
    attribution: str
