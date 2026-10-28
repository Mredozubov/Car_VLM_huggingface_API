from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backend.app.schemas.prediction import ReferenceMatch


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    make: str
    model: str
    generation: str
    year_start: int
    year_end: int
    source_url: str
    license: str
    attribution: str
    embedding: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ReferenceRecord:
        return cls(
            reference_id=str(value["reference_id"]),
            make=str(value["make"]),
            model=str(value["model"]),
            generation=str(value["generation"]),
            year_start=int(value["year_start"]),
            year_end=int(value["year_end"]),
            source_url=str(value.get("source_url", "")),
            license=str(value.get("license", "")),
            attribution=str(value.get("attribution", "")),
            embedding=tuple(float(item) for item in value["embedding"]),
        )


class ReferenceGallery:
    """Small JSONL reference index; replaceable with FAISS when the gallery grows."""

    def __init__(self, records: Iterable[ReferenceRecord] = ()) -> None:
        self.records = tuple(records)

    @classmethod
    def load(cls, path: Path) -> ReferenceGallery:
        if not path.is_file():
            return cls()
        records: list[ReferenceRecord] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(ReferenceRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid reference record at {path}:{line_number}: {error}"
                ) from error
        return cls(records)

    def nearest(
        self,
        embedding: tuple[float, ...],
        *,
        make: str | None = None,
        limit: int = 3,
    ) -> tuple[ReferenceMatch, ...]:
        scored: list[tuple[float, ReferenceRecord]] = []
        for record in self.records:
            if make and record.make.casefold() != make.casefold():
                continue
            similarity = _cosine_similarity(embedding, record.embedding)
            scored.append((similarity, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return tuple(
            ReferenceMatch(
                reference_id=record.reference_id,
                make=record.make,
                model=record.model,
                generation=record.generation,
                year_range=f"{record.year_start}-{record.year_end}",
                similarity=similarity,
                source_url=record.source_url,
                license=record.license,
                attribution=record.attribution,
            )
            for similarity, record in scored[:limit]
        )

    def get(self, reference_id: str) -> ReferenceRecord | None:
        return next(
            (record for record in self.records if record.reference_id == reference_id),
            None,
        )


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
