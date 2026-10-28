from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from backend.app.schemas.prediction import ClassificationResult


@dataclass(frozen=True)
class CachedPrediction:
    classification: ClassificationResult
    image_fingerprint: str
    created_at: float
    bounding_box: tuple[int, int, int, int] | None = None


class PredictionCache:
    """Short-lived metadata cache. It never retains uploaded image pixels."""

    def __init__(self, max_items: int = 200, ttl_seconds: int = 1800) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, CachedPrediction] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, prediction_id: str, value: CachedPrediction) -> None:
        with self._lock:
            self._purge()
            self._items[prediction_id] = value
            self._items.move_to_end(prediction_id)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def get(self, prediction_id: str) -> CachedPrediction | None:
        with self._lock:
            self._purge()
            value = self._items.get(prediction_id)
            if value:
                self._items.move_to_end(prediction_id)
            return value

    def _purge(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        expired = [key for key, value in self._items.items() if value.created_at < cutoff]
        for key in expired:
            self._items.pop(key, None)
