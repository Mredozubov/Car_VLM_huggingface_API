import json
from pathlib import Path

from backend.app.services.reference_store import ReferenceGallery, ReferenceRecord


def test_nearest_reference_filters_by_make() -> None:
    gallery = ReferenceGallery(
        [
            ReferenceRecord(
                reference_id="subaru-1",
                make="Subaru",
                model="Forester",
                generation="Second generation",
                year_start=2003,
                year_end=2008,
                source_url="https://example.test/subaru",
                license="CC BY 4.0",
                attribution="Example",
                embedding=(1.0, 0.0),
            ),
            ReferenceRecord(
                reference_id="ford-1",
                make="Ford",
                model="Escape",
                generation="First generation",
                year_start=2001,
                year_end=2007,
                source_url="https://example.test/ford",
                license="CC BY 4.0",
                attribution="Example",
                embedding=(0.99, 0.01),
            ),
        ]
    )

    matches = gallery.nearest((1.0, 0.0), make="Subaru")

    assert len(matches) == 1
    assert matches[0].reference_id == "subaru-1"
    assert matches[0].similarity == 1.0


def test_loads_jsonl_gallery(tmp_path: Path) -> None:
    path = tmp_path / "references.jsonl"
    path.write_text(
        json.dumps(
            {
                "reference_id": "ref-1",
                "make": "Subaru",
                "model": "Forester",
                "generation": "Second generation",
                "year_start": 2003,
                "year_end": 2008,
                "source_url": "https://example.test/ref-1",
                "license": "CC BY 4.0",
                "attribution": "Example",
                "embedding": [1.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    gallery = ReferenceGallery.load(path)

    assert len(gallery.records) == 1
    assert gallery.records[0].model == "Forester"

