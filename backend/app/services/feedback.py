from __future__ import annotations

import hashlib
import io
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from backend.app.core.config import CONTRIBUTIONS_ROOT, FEEDBACK_DATABASE_PATH
from backend.app.schemas.prediction import VehiclePrediction


@dataclass(frozen=True)
class StoredConfirmation:
    confirmation_id: str
    prediction_id: str
    image_stored: bool
    original_path: str | None
    crop_path: str | None


class FeedbackStore:
    """Local consent-aware store; replaceable by Postgres/object storage later."""

    def __init__(
        self,
        database_path: Path = FEEDBACK_DATABASE_PATH,
        contributions_root: Path = CONTRIBUTIONS_ROOT,
    ) -> None:
        self.database_path = database_path
        self.contributions_root = contributions_root
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS confirmations (
                    confirmation_id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL,
                    selected_rank INTEGER NOT NULL CHECK(selected_rank BETWEEN 1 AND 5),
                    make TEXT NOT NULL,
                    model TEXT NOT NULL,
                    year_range TEXT NOT NULL,
                    user_confidence TEXT NOT NULL,
                    consent_to_training INTEGER NOT NULL,
                    image_sha256 TEXT,
                    original_path TEXT,
                    crop_path TEXT,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )

    def save_confirmation(
        self,
        *,
        prediction_id: str,
        selected_rank: int,
        prediction: VehiclePrediction,
        user_confidence: str,
        consent_to_training: bool,
        original_image: Image.Image | None,
        crop_image: Image.Image | None,
        model_id: str,
    ) -> StoredConfirmation:
        if selected_rank not in range(1, 6):
            raise ValueError("Selected rank must be between 1 and 5.")
        if user_confidence not in {"certain", "likely", "unsure"}:
            raise ValueError("User confidence must be certain, likely, or unsure.")

        confirmation_id = uuid.uuid4().hex
        image_hash: str | None = None
        original_path: Path | None = None
        crop_path: Path | None = None
        if consent_to_training:
            if original_image is None or crop_image is None:
                raise ValueError("Consented image storage requires the original and crop.")
            self.contributions_root.mkdir(parents=True, exist_ok=True)
            original_bytes = _sanitized_jpeg(original_image)
            crop_bytes = _sanitized_jpeg(crop_image)
            image_hash = hashlib.sha256(original_bytes).hexdigest()
            original_path = self.contributions_root / f"{confirmation_id}-original.jpg"
            crop_path = self.contributions_root / f"{confirmation_id}-crop.jpg"
            original_path.write_bytes(original_bytes)
            crop_path.write_bytes(crop_bytes)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO confirmations (
                    confirmation_id, prediction_id, selected_rank, make, model,
                    year_range, user_confidence, consent_to_training, image_sha256,
                    original_path, crop_path, model_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    confirmation_id,
                    prediction_id,
                    selected_rank,
                    prediction.make,
                    prediction.model,
                    prediction.year_range,
                    user_confidence,
                    int(consent_to_training),
                    image_hash,
                    str(original_path) if original_path else None,
                    str(crop_path) if crop_path else None,
                    model_id,
                    datetime.now(UTC).isoformat(),
                ),
            )

        return StoredConfirmation(
            confirmation_id=confirmation_id,
            prediction_id=prediction_id,
            image_stored=consent_to_training,
            original_path=str(original_path) if original_path else None,
            crop_path=str(crop_path) if crop_path else None,
        )


def _sanitized_jpeg(image: Image.Image) -> bytes:
    """Re-encode pixels only, deliberately dropping filename and EXIF metadata."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue()
