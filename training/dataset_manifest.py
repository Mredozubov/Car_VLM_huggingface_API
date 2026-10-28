"""Validated, dataset-neutral records for US-market vehicle training and evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TARGET_YEAR_START = 1980
TARGET_YEAR_END = 2026
VIEWPOINTS = {"front", "front_three_quarter", "side", "rear_three_quarter", "rear", "unknown"}
PART_NAMES = {
    "headlight", "taillight", "grille", "door", "window", "spoiler", "wheel", "mirror"
}


@dataclass(frozen=True)
class ManifestRecord:
    record_id: str
    image_path: str
    make: str
    model: str
    year_start: int
    year_end: int
    generation: str
    market: str
    split: str
    source_dataset: str
    license: str
    viewpoint: str = "unknown"
    parts: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, object]) -> "ManifestRecord":
        raw_parts = item.get("parts", {})
        if not isinstance(raw_parts, dict):
            raise ValueError("parts must be an object of name-to-box mappings")
        parts = {
            str(name): tuple(int(value) for value in box)  # type: ignore[arg-type]
            for name, box in raw_parts.items()
        }
        return cls(
            record_id=str(item["record_id"]).strip(),
            image_path=str(item["image_path"]).strip(),
            make=str(item["make"]).strip(),
            model=str(item["model"]).strip(),
            year_start=int(item["year_start"]),
            year_end=int(item["year_end"]),
            generation=str(item.get("generation", "Unknown generation")).strip(),
            market=str(item.get("market", "US")).strip(),
            split=str(item.get("split", "train")).strip(),
            source_dataset=str(item["source_dataset"]).strip(),
            license=str(item["license"]).strip(),
            viewpoint=str(item.get("viewpoint", "unknown")).strip(),
            parts=parts,
        )

    def to_dict(self) -> dict[str, object]:
        result = dict(self.__dict__)
        result["parts"] = {name: list(box) for name, box in self.parts.items()}
        return result


def load_manifest(path: Path, *, require_images: bool = True) -> tuple[ManifestRecord, ...]:
    records: list[ManifestRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(ManifestRecord.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
    validate_manifest(records, base_directory=path.parent, require_images=require_images)
    return tuple(records)


def validate_manifest(
    records: Iterable[ManifestRecord], *, base_directory: Path, require_images: bool = True
) -> None:
    identifiers: set[str] = set()
    for record in records:
        if not record.record_id or record.record_id in identifiers:
            raise ValueError(f"Missing or duplicate record_id: {record.record_id!r}")
        identifiers.add(record.record_id)
        if not record.make or not record.model:
            raise ValueError(f"Make and model are required for {record.record_id}")
        if not TARGET_YEAR_START <= record.year_start <= record.year_end <= TARGET_YEAR_END:
            raise ValueError(f"Year range outside 1980-2026 for {record.record_id}")
        if record.market.casefold() != "us":
            raise ValueError(f"Non-US record in US manifest: {record.record_id}")
        if record.split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split for {record.record_id}: {record.split}")
        if not record.source_dataset or not record.license:
            raise ValueError(f"Source dataset and license are required for {record.record_id}")
        if record.viewpoint not in VIEWPOINTS:
            raise ValueError(f"Invalid viewpoint for {record.record_id}: {record.viewpoint}")
        for name, box in record.parts.items():
            if name not in PART_NAMES or len(box) != 4:
                raise ValueError(f"Invalid part annotation {name!r} for {record.record_id}")
            left, top, right, bottom = box
            if min(box) < 0 or right <= left or bottom <= top:
                raise ValueError(f"Invalid {name} box for {record.record_id}")
        image_path = Path(record.image_path)
        resolved = image_path if image_path.is_absolute() else base_directory / image_path
        if require_images and not resolved.is_file():
            raise ValueError(f"Image does not exist for {record.record_id}: {resolved}")


def write_manifest(path: Path, records: Iterable[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in records)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()
    records = load_manifest(args.manifest, require_images=not args.allow_missing_images)
    print(f"Manifest valid: {len(records)} records")
    print(f"Test records: {sum(record.split == 'test' for record in records)}")
    print(f"Part-annotated: {sum(bool(record.parts) for record in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
