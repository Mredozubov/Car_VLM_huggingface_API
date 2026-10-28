import pytest
import torch
from PIL import Image

from backend.app.schemas.prediction import VehiclePrediction
from backend.app.services.classifier import (
    _attention_overlay,
    _has_reference_support,
    display_year_range,
    parse_vehicle_label,
    recognition_decision,
)
from backend.app.schemas.prediction import ReferenceMatch


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Audi S5 Coupe 2012", ("Audi", "S5 Coupe", 2012)),
        (
            "Land Rover Range Rover SUV 2012",
            ("Land Rover", "Range Rover SUV", 2012),
        ),
        (
            "2012 Tesla Model S",
            ("Tesla", "Model S", 2012),
        ),
        (
            "Mercedes-Benz C-Class Sedan 2012",
            ("Mercedes-Benz", "C-Class Sedan", 2012),
        ),
    ],
)
def test_parses_vehicle_labels(
    label: str, expected: tuple[str, str, int]
) -> None:
    assert parse_vehicle_label(label) == expected


def test_low_confidence_uses_estimated_year_range() -> None:
    assert display_year_range(2012, 0.40) == "2010-2014 (estimated range)"


def test_high_confidence_keeps_dataset_reference_year() -> None:
    assert display_year_range(2012, 0.90) == "2012"


def test_missing_year_is_explicit() -> None:
    assert display_year_range(None, 0.90) == "Generation/year range unavailable"


def test_prediction_serializes_confidence_as_percentage() -> None:
    prediction = VehiclePrediction(
        make="Subaru",
        model="Forester SUV",
        year_range="2010-2014 (estimated range)",
        confidence=0.87678,
        raw_label="Subaru Forester SUV 2012",
    )

    assert prediction.to_dict()["confidence_percent"] == 87.68
    assert "confidence" not in prediction.to_dict()


def test_attention_overlay_preserves_image_size() -> None:
    image = Image.new("RGB", (80, 40), "blue")
    attention = torch.rand(1, 2, 5, 5)

    overlay = _attention_overlay(image, (attention,))

    assert overlay.size == image.size
    assert overlay.mode == "RGB"


def test_reference_support_requires_same_model_and_similarity() -> None:
    reference = ReferenceMatch(
        reference_id="ref-1",
        make="Subaru",
        model="Forester",
        generation="Second generation",
        year_range="2003-2008",
        similarity=0.91,
        source_url="https://example.test/ref",
        license="CC BY 4.0",
        attribution="Example",
    )

    assert _has_reference_support("Forester SUV", (reference,)) is True
    assert _has_reference_support("Outback Wagon", (reference,)) is False


def test_unknown_policy_rejects_low_score_or_small_margin() -> None:
    assert recognition_decision(0.20, 0.15)[0] is True
    assert recognition_decision(0.70, 0.02)[0] is True
    assert recognition_decision(0.70, 0.20)[0] is False
