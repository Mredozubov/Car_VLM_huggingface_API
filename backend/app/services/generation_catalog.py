from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VehicleGeneration:
    vehicle_id: str
    make: str
    model: str
    generation: str
    year_start: int
    year_end: int
    market: str
    source_url: str
    review_status: str

    @property
    def year_range(self) -> str:
        return f"{self.year_start}-{self.year_end}"


class GenerationCatalog:
    def __init__(self, generations: tuple[VehicleGeneration, ...] = ()) -> None:
        self.generations = generations
        self._validate()

    @classmethod
    def load(cls, path: Path) -> GenerationCatalog:
        if not path.is_file():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Generation catalog must contain a JSON list.")
        generations = tuple(
            VehicleGeneration(
                vehicle_id=str(item["vehicle_id"]),
                make=str(item["make"]),
                model=str(item["model"]),
                generation=str(item["generation"]),
                year_start=int(item["year_start"]),
                year_end=int(item["year_end"]),
                market=str(item.get("market", "US")),
                source_url=str(item.get("source_url", "")),
                review_status=str(item.get("review_status", "pending")),
            )
            for item in payload
        )
        return cls(generations)

    def lookup(
        self, make: str, model: str, reference_year: int | None
    ) -> VehicleGeneration | None:
        if reference_year is None:
            return None
        normalized_make = _normalize_name(make)
        normalized_model = _normalize_model(model)
        candidates = [
            generation
            for generation in self.generations
            if generation.market.casefold() == "us"
            and _normalize_name(generation.make) == normalized_make
            and _normalize_model(generation.model) == normalized_model
            and generation.year_start <= reference_year <= generation.year_end
            and generation.review_status == "verified"
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _validate(self) -> None:
        identifiers: set[str] = set()
        intervals: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        for generation in self.generations:
            if generation.vehicle_id in identifiers:
                raise ValueError(f"Duplicate vehicle_id: {generation.vehicle_id}")
            identifiers.add(generation.vehicle_id)
            if generation.year_start > generation.year_end:
                raise ValueError(
                    f"Invalid year range for {generation.vehicle_id}: "
                    f"{generation.year_start}-{generation.year_end}"
                )
            if generation.review_status not in {"pending", "verified"}:
                raise ValueError(
                    f"Invalid review status for {generation.vehicle_id}: "
                    f"{generation.review_status}"
                )
            if generation.review_status == "verified" and not generation.source_url:
                raise ValueError(
                    f"Verified generation lacks a source URL: {generation.vehicle_id}"
                )
            key = (
                _normalize_name(generation.make),
                _normalize_model(generation.model),
                generation.market.casefold(),
            )
            ranges = intervals.setdefault(key, [])
            for start, end in ranges:
                if generation.year_start <= end and start <= generation.year_end:
                    raise ValueError(
                        f"Overlapping generation ranges for {generation.make} "
                        f"{generation.model}"
                    )
            ranges.append((generation.year_start, generation.year_end))


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def _normalize_model(value: str) -> str:
    body_terms = {
        "cab", "convertible", "coupe", "hatchback", "minivan", "sedan", "suv",
        "van", "wagon",
    }
    return " ".join(
        token for token in _normalize_name(value).split() if token not in body_terms
    )
