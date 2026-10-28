"""Validate that every dataset source has an explicit rights and deployment decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VALID_CATEGORIES = {"images", "metadata"}
VALID_STATUSES = {"approved", "conditional", "research_only", "pending_review", "blocked"}
BOOLEAN_FIELDS = {
    "training_allowed",
    "deployment_allowed",
    "redistribution_allowed",
    "automated_collection_allowed",
    "requires_written_permission",
}


def validate_registry(records: list[dict[str, object]]) -> dict[str, int]:
    identifiers: set[str] = set()
    status_counts = {status: 0 for status in VALID_STATUSES}
    for record in records:
        source_id = str(record.get("source_id", "")).strip()
        if not source_id or source_id in identifiers:
            raise ValueError(f"Missing or duplicate source_id: {source_id!r}")
        identifiers.add(source_id)
        category = str(record.get("category", ""))
        status = str(record.get("status", ""))
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category for {source_id}: {category}")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status for {source_id}: {status}")
        status_counts[status] += 1
        for field in BOOLEAN_FIELDS:
            if not isinstance(record.get(field), bool):
                raise ValueError(f"{field} must be boolean for {source_id}")
        if status in {"blocked", "pending_review"} and record["training_allowed"]:
            raise ValueError(f"Unsafe training permission for {source_id}")
        if status == "research_only" and record["deployment_allowed"]:
            raise ValueError(f"Research-only source cannot permit deployment: {source_id}")
        if record["deployment_allowed"] and not record["training_allowed"] and category == "images":
            raise ValueError(f"Image deployment allowed without training rights: {source_id}")
        if not str(record.get("notes", "")).strip():
            raise ValueError(f"Missing decision notes for {source_id}")
    return {"sources": len(records), **status_counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("training/data/source-registry.json"),
    )
    args = parser.parse_args()
    records = json.loads(args.registry.read_text(encoding="utf-8"))
    metrics = validate_registry(records)
    print(f"Source registry valid: {metrics['sources']} sources")
    for status in sorted(VALID_STATUSES):
        print(f"{status}: {metrics[status]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
