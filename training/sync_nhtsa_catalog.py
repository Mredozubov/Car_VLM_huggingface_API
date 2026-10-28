"""Download make/model/year metadata from the free NHTSA vPIC API.

vPIC's GetModelsForMakeYear operation supports model years after 1995. Curate
1980-1995 separately and merge it during dataset preparation.
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_TEMPLATE = (
    "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/"
    "make/{make}/modelyear/{year}?format=json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--makes-file",
        required=True,
        type=Path,
        help="UTF-8 file containing one US-market make per line",
    )
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/nhtsa-us-catalog.json"),
    )
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def fetch_models(year: int, make: str, delay: float) -> list[dict[str, object]]:
    url = API_TEMPLATE.format(make=urllib.parse.quote(make, safe=""), year=year)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
            time.sleep(max(0.0, delay))
            return parse_results(payload, year=year, make=make, source_url=url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == 2:
                print(f"Warning: skipped {year} {make}: {error}")
                return []
            time.sleep(1.0 * (attempt + 1))
    return []


def parse_results(
    payload: dict[str, object], *, year: int, make: str, source_url: str
) -> list[dict[str, object]]:
    """Remove vPIC's partial-make matches before they enter the taxonomy."""
    records: list[dict[str, object]] = []
    raw_results = payload.get("Results", [])
    if not isinstance(raw_results, list):
        return records
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        canonical_make = str(item.get("Make_Name", make)).strip()
        model = str(item.get("Model_Name", "")).strip()
        if canonical_make.casefold() != make.casefold() or not model:
            continue
        records.append(
            {
                "year": year,
                "make": canonical_make,
                "model": model,
                "market": "US",
                "source": "NHTSA vPIC",
                "source_url": source_url,
            }
        )
    return records


def main() -> int:
    args = parse_args()
    if args.start_year < 1996:
        raise SystemExit("Error: vPIC model-year lookup requires 1996 or later.")
    if args.start_year > args.end_year:
        raise SystemExit("Error: start year must not be after end year.")

    makes = [
        line.strip()
        for line in args.makes_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    output = args.output.resolve()
    records: dict[tuple[int, str, str], dict[str, object]] = {}
    if args.resume and output.is_file():
        for item in json.loads(output.read_text(encoding="utf-8")):
            key = (int(item["year"]), str(item["make"]).casefold(), str(item["model"]).casefold())
            records[key] = item
    completed = {(key[0], key[1]) for key in records}
    jobs = [
        (year, make)
        for year in range(args.start_year, args.end_year + 1)
        for make in makes
        if (year, make.casefold()) not in completed
    ]
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = {
            executor.submit(fetch_models, year, make, args.delay): (year, make)
            for year, make in jobs
        }
        for completed_count, future in enumerate(as_completed(futures), 1):
            for item in future.result():
                key = (int(item["year"]), str(item["make"]).casefold(), str(item["model"]).casefold())
                records[key] = item
            if completed_count % 100 == 0:
                print(f"Completed {completed_count}/{len(jobs)} NHTSA queries")

    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records.values(), key=lambda item: (item["year"], item["make"], item["model"])
    )
    output.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"Saved {len(ordered)} catalog entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
