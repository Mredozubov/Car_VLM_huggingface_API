"""Validate generation IDs, ranges, overlap, sources, and review status."""

import argparse
from pathlib import Path

from backend.app.core.config import GENERATION_CATALOG_PATH
from backend.app.services.generation_catalog import GenerationCatalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=GENERATION_CATALOG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = GenerationCatalog.load(args.catalog)
    verified = sum(
        generation.review_status == "verified" for generation in catalog.generations
    )
    pending = len(catalog.generations) - verified
    print(f"Catalog valid: {len(catalog.generations)} generation records")
    print(f"Verified: {verified}")
    print(f"Pending source review: {pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
