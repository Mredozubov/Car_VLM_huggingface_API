import base64
import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.app.schemas.prediction import ClassificationResult, VehiclePrediction
from backend.app.services.detector import DetectionResult
from backend.app.services.detector import VehicleNotDetectedError
from backend.app.services.feedback import FeedbackStore
from backend.app.services.generation_catalog import GenerationCatalog
from backend.app.services.inference import InferenceOutcome
from backend.app.services.reference_store import ReferenceGallery


class FakeClassifier:
    model_id = "fake-car-model"
    reference_gallery = ReferenceGallery()
    generation_catalog = GenerationCatalog()


class FakeInferenceService:
    classifier = FakeClassifier()

    def predict(self, image: Image.Image) -> InferenceOutcome:
        crop = image.crop((0, 0, image.width, image.height))
        detection = DetectionResult("car", 0.90, (0, 0, image.width, image.height), crop)
        predictions = tuple(
            VehiclePrediction(
                make="Make",
                model=f"Model {rank}",
                year_range="2000-2004",
                confidence=0.5 / rank,
                raw_label=f"Make Model {rank} 2002",
            )
            for rank in range(1, 6)
        )
        classification = ClassificationResult(
            predictions=predictions,
            low_confidence=True,
            message="Test uncertainty",
            model_id=self.classifier.model_id,
            limitations="Test limitations",
            explanation="Test explanation",
        )
        attention = base64.b64encode(b"fake-jpeg").decode("ascii")
        return InferenceOutcome(
            detection, classification, f"data:image/jpeg;base64,{attention}"
        )

    def detect_crop(self, image: Image.Image) -> Image.Image:
        return image.copy()

    def detect_all(self, image: Image.Image) -> tuple[DetectionResult, ...]:
        return (
            DetectionResult("car", 0.90, (0, 0, 16, 24), image.crop((0, 0, 16, 24))),
            DetectionResult("truck", 0.75, (16, 0, 32, 24), image.crop((16, 0, 32, 24))),
        )

    def predict_selection(
        self,
        image: Image.Image,
        bounding_box: tuple[int, int, int, int],
        mask: Image.Image | None = None,
    ) -> InferenceOutcome:
        outcome = self.predict(image.crop(bounding_box))
        return InferenceOutcome(
            DetectionResult(
                "manual painted selection" if mask else "user-selected vehicle",
                1.0,
                bounding_box,
                image.crop(bounding_box),
            ),
            outcome.classification,
            outcome.attention_data_url,
        )


class NoVehicleInferenceService(FakeInferenceService):
    def predict(self, image: Image.Image) -> InferenceOutcome:
        raise VehicleNotDetectedError("No vehicle detected with at least 55% confidence.")

    def detect_all(self, image: Image.Image) -> tuple[DetectionResult, ...]:
        return ()


def _jpeg_bytes(color: str = "blue") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        inference_service=FakeInferenceService(),
        feedback_store=FeedbackStore(
            tmp_path / "feedback.sqlite3", tmp_path / "contributions"
        ),
    )
    return TestClient(app)


def test_health_and_prediction(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        health = client.get("/api/v1/health")
        response = client.post(
            "/api/v1/predict",
            files={"file": ("car.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert health.status_code == 200
    assert health.json()["models_loaded"] is True
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction_id"].startswith("pred_")
    assert len(payload["classification"]["predictions"]) == 5
    assert payload["attention_image_data_url"].startswith("data:image/jpeg;base64,")


def test_rejects_invalid_upload(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("car.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_api_rejects_oversized_upload(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("car.jpg", b"x" * (10 * 1024 * 1024 + 1), "image/jpeg")},
        )

    assert response.status_code == 400
    assert "Maximum size is 10 MB" in response.json()["detail"]


def test_api_explains_when_no_vehicle_is_detected(tmp_path: Path) -> None:
    app = create_app(
        inference_service=NoVehicleInferenceService(),
        feedback_store=FeedbackStore(tmp_path / "feedback.sqlite3", tmp_path / "contributions"),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("scene.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 422
    assert "No vehicle detected" in response.json()["detail"]


def test_local_frontend_cors_preflight(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.options(
            "/api/v1/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_detect_returns_all_vehicle_choices(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/detect",
            files={"file": ("cars.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["image_width"] == 32
    assert len(response.json()["detections"]) == 2
    assert response.json()["detections"][1]["vehicle_type"] == "truck"


def test_predict_accepts_user_selected_box(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            data={
                "selection_left": "4",
                "selection_top": "3",
                "selection_right": "28",
                "selection_bottom": "20",
            },
            files={"file": ("cars.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["detection"]["bounding_box"] == [4, 3, 28, 20]
    assert response.json()["detection"]["vehicle_type"] == "user-selected vehicle"


def test_feedback_without_consent_retains_no_image(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        prediction = client.post(
            "/api/v1/predict",
            files={"file": ("car.jpg", _jpeg_bytes(), "image/jpeg")},
        ).json()
        response = client.post(
            "/api/v1/feedback",
            data={
                "prediction_id": prediction["prediction_id"],
                "selected_rank": "1",
                "user_confidence": "certain",
                "consent_to_training": "false",
            },
        )

    assert response.status_code == 200
    assert response.json()["image_stored"] is False
    assert not (tmp_path / "contributions").exists()


def test_feedback_with_consent_requires_matching_image(tmp_path: Path) -> None:
    original = _jpeg_bytes("blue")
    with _client(tmp_path) as client:
        prediction = client.post(
            "/api/v1/predict",
            files={"file": ("car.jpg", original, "image/jpeg")},
        ).json()
        mismatch = client.post(
            "/api/v1/feedback",
            data={
                "prediction_id": prediction["prediction_id"],
                "selected_rank": "1",
                "consent_to_training": "true",
            },
            files={"file": ("car.jpg", _jpeg_bytes("red"), "image/jpeg")},
        )
        accepted = client.post(
            "/api/v1/feedback",
            data={
                "prediction_id": prediction["prediction_id"],
                "selected_rank": "1",
                "consent_to_training": "true",
            },
            files={"file": ("car.jpg", original, "image/jpeg")},
        )

    assert mismatch.status_code == 409
    assert accepted.status_code == 200
    assert accepted.json()["image_stored"] is True
    assert len(list((tmp_path / "contributions").glob("*.jpg"))) == 2


def test_unknown_reference_returns_404(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/api/v1/references/missing")

    assert response.status_code == 404
