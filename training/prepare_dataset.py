"""Convert a reviewed CSV label export into the normalized MotorLens JSONL manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.dataset_manifest import ManifestRecord, validate_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=("VMMRdb", "CompCars", "Vehicle-1M", "consented"))
    parser.add_argument("--license", required=True, help="Reviewed license identifier or terms reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[ManifestRecord] = []
    with args.labels.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            image = (args.images_root / row["image_path"]).resolve()
            parts = json.loads(row.get("parts_json") or "{}")
            records.append(
                ManifestRecord.from_dict(
                    {
                        "record_id": row.get("record_id") or f"{args.dataset.casefold()}-{index:08d}",
                        "image_path": str(image),
                        "make": row["make"],
                        "model": row["model"],
                        "year_start": row.get("year_start") or row["year"],
                        "year_end": row.get("year_end") or row["year"],
                        "generation": row.get("generation") or "Unknown generation",
                        "market": row.get("market") or "US",
                        "split": row.get("split") or "train",
                        "source_dataset": args.dataset,
                        "license": args.license,
                        "viewpoint": row.get("viewpoint") or "unknown",
                        "parts": parts,
                    }
                )
            )
    validate_manifest(records, base_directory=args.output.parent)
    write_manifest(args.output, records)
    print(f"Wrote {len(records)} normalized records to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
