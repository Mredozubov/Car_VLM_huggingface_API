from __future__ import annotations

import base64
import hashlib
import io
import threading
from dataclasses import dataclass

from PIL import Image

from backend.app.schemas.prediction import ClassificationResult
from backend.app.services.classifier import HuggingFaceVehicleClassifier
from backend.app.services.detector import DetectionResult, TorchvisionVehicleDetector


@dataclass(frozen=True)
class InferenceOutcome:
    detection: DetectionResult
    classification: ClassificationResult
    attention_data_url: str


class InferenceService:
    """Owns models loaded once and serializes access for predictable local inference."""

    def __init__(
        self,
        detector: TorchvisionVehicleDetector | None = None,
        classifier: HuggingFaceVehicleClassifier | None = None,
    ) -> None:
        self.detector = detector or TorchvisionVehicleDetector()
        self.classifier = classifier or HuggingFaceVehicleClassifier()
        self._lock = threading.Lock()

    def predict(self, image: Image.Image) -> InferenceOutcome:
        with self._lock:
            detection = self.detector.detect_best(image)
            analysis = self.classifier.analyze(detection.crop)
        buffer = io.BytesIO()
        analysis.attention_overlay.save(buffer, format="JPEG", quality=85)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )
        return InferenceOutcome(detection, analysis.result, data_url)

    def detect_all(self, image: Image.Image) -> tuple[DetectionResult, ...]:
        with self._lock:
            return self.detector.detect_all(image)

    def predict_selection(
        self,
        image: Image.Image,
        bounding_box: tuple[int, int, int, int],
        mask: Image.Image | None = None,
    ) -> InferenceOutcome:
        rgb_image = image.convert("RGB")
        left, top, right, bottom = _validated_box(bounding_box, rgb_image.size)
        crop = rgb_image.crop((left, top, right, bottom))
        label = "user-selected vehicle"
        if mask is not None:
            if mask.size != rgb_image.size:
                raise ValueError("Selection mask dimensions must match the uploaded image.")
            mask_crop = mask.convert("L").crop((left, top, right, bottom))
            if mask_crop.getbbox() is None:
                raise ValueError("Paint over the vehicle before analyzing it.")
            neutral = Image.new("RGB", crop.size, (127, 127, 127))
            crop = Image.composite(crop, neutral, mask_crop)
            label = "manual painted selection"

        detection = DetectionResult(label, 1.0, (left, top, right, bottom), crop)
        with self._lock:
            analysis = self.classifier.analyze(crop)
        buffer = io.BytesIO()
        analysis.attention_overlay.save(buffer, format="JPEG", quality=85)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )
        return InferenceOutcome(detection, analysis.result, data_url)

    def detect_crop(self, image: Image.Image) -> Image.Image:
        with self._lock:
            return self.detector.detect_best(image).crop


def image_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validated_box(
    box: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = box
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    if right - left < 12 or bottom - top < 12:
        raise ValueError("The selected area is too small. Highlight more of the vehicle.")
    return left, top, right, bottom
