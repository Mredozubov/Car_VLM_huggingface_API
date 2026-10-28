import argparse
import sys
from pathlib import Path

from backend.app.core.config import DEFAULT_DETECTION_THRESHOLD
from backend.app.services.detector import (
    TorchvisionVehicleDetector,
    VehicleNotDetectedError,
)
from backend.app.utils.images import ImageValidationError, load_validated_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the most likely vehicle in an image and save its crop."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source image path")
    parser.add_argument("--output", required=True, type=Path, help="Crop output path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
        help="Minimum detection confidence from 0 to 1 (default: 0.55)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        image = load_validated_image(args.input)
        detector = TorchvisionVehicleDetector(threshold=args.threshold)
        result = detector.detect_best(image)
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.crop.save(output_path)
    except (ImageValidationError, VehicleNotDetectedError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: Could not save the crop: {error}", file=sys.stderr)
        return 1

    print(f"Detected {result.label} with {result.confidence:.1%} confidence.")
    print(f"Crop saved to: {output_path}")
    print(f"Bounding box: {result.bounding_box}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

