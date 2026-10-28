import argparse
import json
import sys
import uuid
from pathlib import Path

from backend.app.core.config import DEFAULT_DETECTION_THRESHOLD
from backend.app.services.classifier import HuggingFaceVehicleClassifier
from backend.app.services.detector import (
    TorchvisionVehicleDetector,
    VehicleNotDetectedError,
)
from backend.app.services.feedback import FeedbackStore
from backend.app.utils.images import ImageValidationError, load_validated_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a vehicle and return its top five identities."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source image path")
    parser.add_argument(
        "--crop-output",
        type=Path,
        help="Optional path at which to save the detected vehicle crop",
    )
    parser.add_argument(
        "--attention-output",
        type=Path,
        help="Optional path at which to save the transformer attention overlay",
    )
    parser.add_argument(
        "--confirm",
        type=int,
        choices=range(1, 6),
        metavar="1-5",
        help="Confirm one of the five ranked predictions",
    )
    parser.add_argument(
        "--user-confidence",
        choices=("certain", "likely", "unsure"),
        default="certain",
        help="How sure the user is about the confirmation",
    )
    parser.add_argument(
        "--contribute",
        action="store_true",
        help="With --confirm, explicitly consent to storing sanitized images for review",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
        help="Minimum detection confidence from 0 to 1 (default: 0.55)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.contribute and args.confirm is None:
        print("Error: --contribute requires --confirm 1-5.", file=sys.stderr)
        return 1
    try:
        image = load_validated_image(args.input)
        detection = TorchvisionVehicleDetector(
            threshold=args.threshold
        ).detect_best(image)
        classifier = HuggingFaceVehicleClassifier()
        analysis = classifier.analyze(detection.crop)
        classification = analysis.result

        if args.crop_output:
            crop_path = args.crop_output.resolve()
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            detection.crop.save(crop_path)
        if args.attention_output:
            attention_path = args.attention_output.resolve()
            attention_path.parent.mkdir(parents=True, exist_ok=True)
            analysis.attention_overlay.save(attention_path)
    except (ImageValidationError, VehicleNotDetectedError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: Could not load a model or save the crop: {error}", file=sys.stderr)
        return 1

    prediction_id = f"pred_{uuid.uuid4().hex}"
    output: dict[str, object] = {
        "prediction_id": prediction_id,
        "detection": {
            "vehicle_type": detection.label,
            "confidence_percent": round(detection.confidence * 100, 2),
            "bounding_box": detection.bounding_box,
        },
        "classification": classification.to_dict(),
    }
    if args.confirm is not None:
        selected = classification.predictions[args.confirm - 1]
        confirmation = FeedbackStore().save_confirmation(
            prediction_id=prediction_id,
            selected_rank=args.confirm,
            prediction=selected,
            user_confidence=args.user_confidence,
            consent_to_training=args.contribute,
            original_image=image,
            crop_image=detection.crop,
            model_id=classification.model_id,
        )
        output["confirmation"] = {
            "confirmation_id": confirmation.confirmation_id,
            "selected_rank": args.confirm,
            "image_stored": confirmation.image_stored,
            "consent_to_training": args.contribute,
        }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
