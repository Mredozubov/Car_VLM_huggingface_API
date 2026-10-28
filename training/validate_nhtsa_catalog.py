"""Validate NHTSA taxonomy scope, uniqueness, source provenance, and year coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_catalog(
    records: list[dict[str, object]], allowed_makes: set[str], start_year: int, end_year: int
) -> dict[str, int]:
    seen: set[tuple[int, str, str]] = set()
    makes: set[str] = set()
    models: set[tuple[str, str]] = set()
    for item in records:
        year = int(item["year"])
        make = str(item["make"]).strip()
        model = str(item["model"]).strip()
        if not start_year <= year <= end_year:
            raise ValueError(f"Out-of-range year: {year}")
        if make.casefold() not in allowed_makes:
            raise ValueError(f"Unexpected partial make match: {make}")
        if not model or not str(item.get("source_url", "")).startswith("https://vpic.nhtsa.dot.gov/"):
            raise ValueError(f"Missing model or NHTSA provenance: {year} {make} {model}")
        key = (year, make.casefold(), model.casefold())
        if key in seen:
            raise ValueError(f"Duplicate taxonomy record: {year} {make} {model}")
        seen.add(key)
        makes.add(make.casefold())
        models.add((make.casefold(), model.casefold()))
    return {"records": len(records), "makes": len(makes), "make_models": len(models)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("training/data/nhtsa-us-catalog.json"))
    parser.add_argument("--makes-file", type=Path, default=Path("training/data/us_makes.txt"))
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()
    makes = {
        line.strip().casefold()
        for line in args.makes_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    records = json.loads(args.catalog.read_text(encoding="utf-8"))
    metrics = validate_catalog(records, makes, args.start_year, args.end_year)
    print(f"NHTSA catalog valid: {metrics['records']} records")
    print(f"Makes with results: {metrics['makes']}")
    print(f"Unique make/model names: {metrics['make_models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
