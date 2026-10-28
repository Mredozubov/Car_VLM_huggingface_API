import sqlite3
from pathlib import Path

from PIL import Image

from backend.app.schemas.prediction import VehiclePrediction
from backend.app.services.feedback import FeedbackStore


def _prediction() -> VehiclePrediction:
    return VehiclePrediction(
        make="Subaru",
        model="Forester",
        year_range="2003-2008",
        confidence=0.80,
        raw_label="Subaru Forester",
    )


def test_confirmation_without_consent_stores_no_images(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3", tmp_path / "images")

    saved = store.save_confirmation(
        prediction_id="pred-1",
        selected_rank=1,
        prediction=_prediction(),
        user_confidence="certain",
        consent_to_training=False,
        original_image=Image.new("RGB", (20, 20), "blue"),
        crop_image=Image.new("RGB", (10, 10), "blue"),
        model_id="test-model",
    )

    assert saved.image_stored is False
    assert saved.original_path is None
    assert not (tmp_path / "images").exists()


def test_consent_stores_sanitized_images_pending_review(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3", tmp_path / "images")
    original = Image.new("RGB", (20, 20), "blue")
    original.info["exif"] = b"private metadata"

    saved = store.save_confirmation(
        prediction_id="pred-2",
        selected_rank=2,
        prediction=_prediction(),
        user_confidence="likely",
        consent_to_training=True,
        original_image=original,
        crop_image=Image.new("RGB", (10, 10), "blue"),
        model_id="test-model",
    )

    assert saved.image_stored is True
    assert saved.original_path is not None
    assert saved.crop_path is not None
    with Image.open(saved.original_path) as stored:
        assert stored.getexif() == {}
    with sqlite3.connect(tmp_path / "feedback.sqlite3") as connection:
        row = connection.execute(
            "SELECT consent_to_training, review_status FROM confirmations"
        ).fetchone()
    assert row == (1, "pending")

