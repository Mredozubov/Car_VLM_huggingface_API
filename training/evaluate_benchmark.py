"""Run or score a frozen MotorLens benchmark without mixing it into training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from backend.app.services.detector import VehicleNotDetectedError
from training.dataset_manifest import ManifestRecord, load_manifest

YEAR_RANGE = re.compile(r"(19\d{2}|20\d{2})(?:-(19\d{2}|20\d{2}))?")


def normalize(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def score_predictions(
    records: tuple[ManifestRecord, ...], predictions: dict[str, dict[str, object]]
) -> dict[str, float | int]:
    test_records = tuple(record for record in records if record.split == "test")
    totals = {
        "samples": len(test_records), "detected": 0, "answered": 0,
        "make_top1": 0, "model_top1": 0, "model_top5": 0, "year_overlap_top1": 0,
    }
    for record in test_records:
        result = predictions.get(record.record_id)
        if not result or not result.get("detected", False):
            continue
        totals["detected"] += 1
        if result.get("abstained", False):
            continue
        totals["answered"] += 1
        candidates = result.get("predictions", [])
        if not isinstance(candidates, list) or not candidates:
            continue
        first = candidates[0]
        if not isinstance(first, dict):
            continue
        if normalize(str(first.get("make", ""))) == normalize(record.make):
            totals["make_top1"] += 1
        expected_model = normalize(record.model)
        if normalize(str(first.get("model", ""))) == expected_model:
            totals["model_top1"] += 1
        if any(isinstance(item, dict) and normalize(str(item.get("model", ""))) == expected_model for item in candidates[:5]):
            totals["model_top5"] += 1
        match = YEAR_RANGE.search(str(first.get("year_range", "")))
        if match:
            predicted_start = int(match.group(1))
            predicted_end = int(match.group(2) or match.group(1))
            if predicted_start <= record.year_end and record.year_start <= predicted_end:
                totals["year_overlap_top1"] += 1

    samples = max(1, totals["samples"])
    detected = max(1, totals["detected"])
    answered = max(1, totals["answered"])
    return {
        **totals,
        "detection_rate": totals["detected"] / samples,
        "answer_rate": totals["answered"] / samples,
        "make_top1_accuracy": totals["make_top1"] / answered,
        "model_top1_accuracy": totals["model_top1"] / answered,
        "model_top5_accuracy": totals["model_top5"] / answered,
        "year_overlap_accuracy": totals["year_overlap_top1"] / answered,
        "detection_denominator": detected,
    }


def run_model(records: tuple[ManifestRecord, ...], manifest_path: Path) -> dict[str, dict[str, object]]:
    from backend.app.services.inference import InferenceService

    service = InferenceService()
    results: dict[str, dict[str, object]] = {}
    for index, record in enumerate((item for item in records if item.split == "test"), 1):
        path = Path(record.image_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        try:
            with Image.open(path) as image:
                outcome = service.predict(image.convert("RGB"))
            payload = outcome.classification.to_dict()
            results[record.record_id] = {
                "detected": True,
                "abstained": payload["abstained"],
                "predictions": payload["predictions"],
            }
        except VehicleNotDetectedError:
            results[record.record_id] = {"detected": False, "abstained": True, "predictions": []}
        print(f"Evaluated {index}: {record.record_id}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="Existing prediction JSON; omit to run the models")
    parser.add_argument("--output", type=Path, default=Path("training/results/benchmark.json"))
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8")) if args.predictions else run_model(records, args.manifest)
    metrics = score_predictions(records, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": metrics, "predictions": predictions}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
