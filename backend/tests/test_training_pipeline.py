import json
from pathlib import Path

import pytest
from PIL import Image

from training.dataset_manifest import ManifestRecord, load_manifest, write_manifest
from training.evaluate_benchmark import score_predictions
from training.sync_nhtsa_catalog import parse_results
from training.part_aware_model import PartAwareFusion
from training.validate_source_registry import validate_registry
import torch


def _record(image_path: str, *, split: str = "test") -> ManifestRecord:
    return ManifestRecord(
        record_id="known-camry",
        image_path=image_path,
        make="Toyota",
        model="Camry Sedan",
        year_start=2012,
        year_end=2014,
        generation="XV50",
        market="US",
        split=split,
        source_dataset="consented",
        license="owner-consent-v1",
        viewpoint="front_three_quarter",
        parts={"headlight": (2, 2, 8, 6), "wheel": (4, 8, 12, 15)},
    )


def test_manifest_round_trip_with_part_annotations(tmp_path: Path) -> None:
    Image.new("RGB", (20, 16), "blue").save(tmp_path / "car.jpg")
    path = tmp_path / "benchmark.jsonl"
    write_manifest(path, [_record("car.jpg")])

    records = load_manifest(path)

    assert records[0].viewpoint == "front_three_quarter"
    assert records[0].parts["headlight"] == (2, 2, 8, 6)


def test_manifest_rejects_out_of_scope_year(tmp_path: Path) -> None:
    record = _record("missing.jpg")
    payload = record.to_dict()
    payload["year_start"] = 1979
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside 1980-2026"):
        load_manifest(path, require_images=False)


def test_benchmark_scores_top_one_top_five_and_abstention() -> None:
    records = (
        _record("car.jpg"),
        ManifestRecord(**{**_record("other.jpg").__dict__, "record_id": "unknown-car"}),
    )
    predictions = {
        "known-camry": {
            "detected": True,
            "abstained": False,
            "predictions": [
                {"make": "Toyota", "model": "Corolla Sedan", "year_range": "2012"},
                {"make": "Toyota", "model": "Camry Sedan", "year_range": "2010-2014"},
            ],
        },
        "unknown-car": {"detected": True, "abstained": True, "predictions": []},
    }

    metrics = score_predictions(records, predictions)

    assert metrics["answer_rate"] == 0.5
    assert metrics["make_top1_accuracy"] == 1.0
    assert metrics["model_top1_accuracy"] == 0.0
    assert metrics["model_top5_accuracy"] == 1.0


def test_nhtsa_parser_removes_partial_make_matches() -> None:
    records = parse_results(
        {
            "Results": [
                {"Make_Name": "RAM", "Model_Name": "1500"},
                {"Make_Name": "RAMPAGE TRAILERS", "Model_Name": "UTILITY"},
            ]
        },
        year=2024,
        make="Ram",
        source_url="https://example.test/nhtsa",
    )

    assert [record["model"] for record in records] == ["1500"]


def test_part_aware_fusion_ignores_hidden_parts() -> None:
    model = PartAwareFusion(8, 12, make_classes=3, model_classes=5, generation_classes=4)
    output = model(
        torch.rand(2, 8),
        torch.rand(2, 3, 8),
        torch.tensor([[0, 1, 6], [2, 4, 7]]),
        torch.tensor([[True, False, True], [False, False, False]]),
        torch.tensor([1, 5]),
    )

    assert output["model_logits"].shape == (2, 5)
    assert output["part_weights"][0, 1].item() == 0.0
    assert output["part_weights"][1].sum().item() == 0.0


def test_source_registry_blocks_research_only_deployment() -> None:
    record = {
        "source_id": "research-set",
        "name": "Research set",
        "category": "images",
        "status": "research_only",
        "training_allowed": True,
        "deployment_allowed": True,
        "redistribution_allowed": False,
        "automated_collection_allowed": False,
        "requires_written_permission": True,
        "terms_url": "https://example.test/terms",
        "notes": "Research only.",
    }

    with pytest.raises(ValueError, match="Research-only source"):
        validate_registry([record])


def test_checked_in_source_registry_is_valid() -> None:
    registry = Path("training/data/source-registry.json")
    metrics = validate_registry(json.loads(registry.read_text(encoding="utf-8")))

    assert metrics["sources"] >= 9
    assert metrics["blocked"] >= 2
