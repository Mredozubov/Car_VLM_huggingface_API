import json
from pathlib import Path

import pytest

from backend.app.services.generation_catalog import GenerationCatalog


def _write_catalog(path: Path, records: list[dict[str, object]]) -> GenerationCatalog:
    path.write_text(json.dumps(records), encoding="utf-8")
    return GenerationCatalog.load(path)


def test_lookup_normalizes_body_style_and_requires_verified(tmp_path: Path) -> None:
    catalog = _write_catalog(
        tmp_path / "generations.json",
        [
            {
                "vehicle_id": "subaru_forester_gen2_us",
                "make": "Subaru",
                "model": "Forester",
                "generation": "Second generation",
                "year_start": 2003,
                "year_end": 2008,
                "market": "US",
                "source_url": "https://example.test/source",
                "review_status": "verified",
            },
            {
                "vehicle_id": "subaru_forester_gen3_us",
                "make": "Subaru",
                "model": "Forester",
                "generation": "Third generation",
                "year_start": 2009,
                "year_end": 2013,
                "market": "US",
                "source_url": "",
                "review_status": "pending",
            },
        ],
    )

    match = catalog.lookup("Subaru", "Forester SUV", 2006)

    assert match is not None
    assert match.generation == "Second generation"
    assert catalog.lookup("Subaru", "Forester SUV", 2012) is None


def test_rejects_overlapping_ranges(tmp_path: Path) -> None:
    records = [
        {
            "vehicle_id": f"vehicle-{index}",
            "make": "Example",
            "model": "Model",
            "generation": f"Generation {index}",
            "year_start": start,
            "year_end": end,
            "market": "US",
            "review_status": "pending",
        }
        for index, (start, end) in enumerate(((2000, 2005), (2005, 2010)), 1)
    ]

    with pytest.raises(ValueError, match="Overlapping"):
        _write_catalog(tmp_path / "generations.json", records)


def test_verified_entry_requires_source(tmp_path: Path) -> None:
    record = {
        "vehicle_id": "example_model_gen1_us",
        "make": "Example",
        "model": "Model",
        "generation": "First generation",
        "year_start": 2000,
        "year_end": 2004,
        "market": "US",
        "source_url": "",
        "review_status": "verified",
    }

    with pytest.raises(ValueError, match="lacks a source URL"):
        _write_catalog(tmp_path / "generations.json", [record])

