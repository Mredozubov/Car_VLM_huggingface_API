"""Curator-only command for adding a licensed, verified gallery reference."""

import argparse
import json
import uuid
from pathlib import Path

from backend.app.core.config import REFERENCE_GALLERY_PATH
from backend.app.services.classifier import HuggingFaceVehicleClassifier
from backend.app.services.detector import TorchvisionVehicleDetector
from backend.app.utils.images import load_validated_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--make", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--year-start", required=True, type=int)
    parser.add_argument("--year-end", required=True, type=int)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--attribution", required=True)
    parser.add_argument("--gallery", type=Path, default=REFERENCE_GALLERY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.year_start > args.year_end:
        raise SystemExit("Error: --year-start must not be after --year-end.")
    image = load_validated_image(args.input)
    crop = TorchvisionVehicleDetector().detect_best(image).crop
    analysis = HuggingFaceVehicleClassifier().analyze(crop)
    record = {
        "reference_id": f"ref_{uuid.uuid4().hex}",
        "make": args.make.strip(),
        "model": args.model.strip(),
        "generation": args.generation.strip(),
        "year_start": args.year_start,
        "year_end": args.year_end,
        "source_url": args.source_url.strip(),
        "license": args.license.strip(),
        "attribution": args.attribution.strip(),
        "embedding": list(analysis.embedding),
    }
    gallery_path = args.gallery.resolve()
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    with gallery_path.open("a", encoding="utf-8") as gallery:
        gallery.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"Added verified reference {record['reference_id']} to {gallery_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

