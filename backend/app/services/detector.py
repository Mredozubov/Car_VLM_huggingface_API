from dataclasses import dataclass
from typing import Protocol

import torch
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import pil_to_tensor

from backend.app.core.config import CROP_PADDING_RATIO, DEFAULT_DETECTION_THRESHOLD

ROAD_VEHICLE_LABELS = frozenset({"car", "motorcycle", "bus", "truck"})


class VehicleNotDetectedError(RuntimeError):
    """Raised when no road vehicle passes the requested confidence threshold."""


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    crop: Image.Image


class VehicleDetector(Protocol):
    def detect_best(self, image: Image.Image) -> DetectionResult:
        """Return the best vehicle detection and its crop."""


class TorchvisionVehicleDetector:
    def __init__(
        self,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        padding_ratio: float = CROP_PADDING_RATIO,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Detection threshold must be between 0 and 1.")
        if not 0.0 <= padding_ratio <= 0.5:
            raise ValueError("Padding ratio must be between 0 and 0.5.")

        self.threshold = threshold
        self.padding_ratio = padding_ratio
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def detect_all(self, image: Image.Image) -> tuple[DetectionResult, ...]:
        rgb_image = image.convert("RGB")
        tensor = pil_to_tensor(rgb_image).float().div(255).to(self.device)
        prediction = self.model([tensor])[0]

        detections: list[DetectionResult] = []
        for box, label_id, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            confidence = float(score.item())
            if confidence < self.threshold:
                break
            label = self.categories[int(label_id.item())]
            if label not in ROAD_VEHICLE_LABELS:
                continue

            bounding_box = self._padded_box(
                tuple(float(value) for value in box.tolist()), rgb_image.size
            )
            detections.append(
                DetectionResult(
                    label=label,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    crop=rgb_image.crop(bounding_box),
                )
            )

        return tuple(detections)

    def detect_best(self, image: Image.Image) -> DetectionResult:
        detections = self.detect_all(image)
        if detections:
            return detections[0]
        raise VehicleNotDetectedError(
            f"No vehicle detected with at least {self.threshold:.0%} confidence."
        )

    def _padded_box(
        self,
        box: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        image_width, image_height = image_size
        pad_x = (right - left) * self.padding_ratio
        pad_y = (bottom - top) * self.padding_ratio
        return (
            max(0, round(left - pad_x)),
            max(0, round(top - pad_y)),
            min(image_width, round(right + pad_x)),
            min(image_height, round(bottom + pad_y)),
        )
