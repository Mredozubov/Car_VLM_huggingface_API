# Training and Phase 6 evaluation

The application runs without training by using the Stanford Cars ViT baseline. This folder provides the repeatable path for measuring that baseline, building the US taxonomy, normalizing licensed datasets, adding part annotations, and evaluating a future replacement model.

Read `training/DATASET_STRATEGY.md` before acquiring images. Validate the machine-readable source decisions with:

```powershell
python -m training.validate_source_registry
```

## 1. Build the 1996-2026 NHTSA catalog

From the repository root in an activated PowerShell environment:

```powershell
python -m training.sync_nhtsa_catalog --makes-file training/data/us_makes.txt --start-year 1996 --end-year 2026 --output training/data/nhtsa-us-catalog.json --workers 4 --delay 0.1 --resume
```

The command queries the free NHTSA vPIC API, retries temporary failures, removes duplicates, and writes source URLs with every record. `--resume` preserves existing results. NHTSA supplies taxonomy metadata, not training photos. Curate 1980-1995 separately in `training/data/legacy_us_catalog.csv` using authoritative sources.

Validate the downloaded catalog:

```powershell
python -m training.validate_nhtsa_catalog --catalog training/data/nhtsa-us-catalog.json --makes-file training/data/us_makes.txt
```

## 2. Create the frozen benchmark

Copy `training/data/dataset-labels.example.csv`, replace the example with reviewed images whose identities are known, and keep those images out of all training splits. A useful first benchmark should balance decades, body styles, common makes, viewpoints, and difficult conditions.

Convert reviewed CSV labels to the normalized manifest:

```powershell
python -m training.prepare_dataset --labels training/data/benchmark-labels.csv --images-root training/data/benchmark-images --dataset consented --license owner-consent-v1 --output training/data/benchmark.jsonl
```

Validate it:

```powershell
python -m training.dataset_manifest training/data/benchmark.jsonl
```

Run the real detector and classifier and save reproducible metrics:

```powershell
python -m training.evaluate_benchmark --manifest training/data/benchmark.jsonl --output training/results/baseline.json
```

Reported metrics include detection rate, answer rate after unknown rejection, top-1 make accuracy, top-1 and top-5 model accuracy, and year-range overlap.

## 3. Normalize a licensed image dataset

Do not copy a dataset into this repository. Download it into a separate local data directory after reviewing and accepting its official terms. Export or map its labels to the example CSV columns, then run:

```powershell
python -m training.prepare_dataset --labels C:\path\to\reviewed-labels.csv --images-root C:\path\to\dataset-images --dataset VMMRdb --license "reviewed-license-id" --output training/data/vmmrdb.jsonl
```

Use `CompCars` or `Vehicle-1M` for `--dataset` when applicable. Records outside the US market or 1980-2026 target range are rejected rather than silently contaminating training.

## Part-aware records

Each manifest record can include `viewpoint` and `parts`. Supported viewpoints are front, front three-quarter, side, rear three-quarter, rear, and unknown. Supported part boxes are headlight, taillight, grille, door, window, spoiler, wheel, and mirror. These annotations are the input contract for the later part-localization and feature-fusion model; the current classifier does not pretend fixed image regions are named parts.

`training/part_aware_model.py` contains the trainable fusion head. It combines a whole-vehicle embedding, visible named-part embeddings, and viewing direction, produces separate make/model/generation logits, and returns part weights that can later support evidence statements. It remains disconnected from production inference until a licensed part-annotated model passes the frozen benchmark.

## Leakage rule

Never allow the same image, near-duplicate image, or same photographed vehicle sequence to appear in both training and test data. Keep the benchmark frozen so before/after accuracy comparisons remain meaningful.
