# MotorLens dataset strategy

## Objective

Build reliable US-market make, model, and generation recognition for model years 2000-2010 first, then expand toward 1980-2026. The production model may train only on sources whose terms permit the intended use. Research-only datasets may be used for experiments, but their images or derived weights must not silently enter a public or commercial deployment.

This document is an engineering policy, not legal advice. When a source's rights are unclear, its status is `pending_review` or `blocked` in `training/data/source-registry.json`.

## Dataset lanes

### Production lane

Eligible after provenance validation:

- Photos taken by the project owner.
- User-contributed photos with explicit training consent.
- Dealership or photographer collections covered by a written machine-learning license.
- OEM or media assets only when their specific terms explicitly allow model training and deployment.
- Public datasets only when the official license permits the intended deployed use.

### Research-only lane

- CompCars: non-commercial research only.
- Vehicle-1M: academic use only and no commercial use.
- Any source agreement that restricts use to research, evaluation, or one organization.

Research-only models must be labeled and stored separately. They can prove whether an architecture works, but they are not production artifacts.

### Metadata lane

- NHTSA vPIC make/model/year records.
- Manually verified generation boundaries and citations.

Metadata defines labels; it does not provide visual training examples.

### Quarantine lane

- Facebook Marketplace images without Meta authorization and photographer permission.
- Dealership listing images collected without a written license.
- Search-engine thumbnails.
- Images with unknown authorship or unclear redistribution/training rights.

Quarantined material is not downloaded, trained on, benchmarked with, or copied into the repository.

## Facebook Marketplace policy

Do not scrape Facebook Marketplace. Meta's automated-data terms require express written permission, and accepting those terms alone is not permission. Public visibility also does not establish that the seller owns every image or licenses it for machine-learning use.

A Marketplace-origin photo is acceptable only when all of the following are true:

1. The seller or photographer submits the original image directly rather than MotorLens collecting it from Facebook.
2. They affirm that they own or control the image rights.
3. They explicitly consent to model training and the intended deployment use.
4. The project records the consent version, submission date, source, and allowed uses.
5. Faces, license plates, location information, EXIF data, listing text, and account identifiers are removed or excluded.

## Dealership policy

Dealership imagery is a strong source because a listing commonly contains front, rear, side, three-quarter, interior, badge, light, and wheel views of the same physical vehicle. Use it only through a direct agreement.

The agreement should state that the dealership or its photography vendor:

- Owns or controls the supplied photographs.
- Grants the right to copy, transform, annotate, and use them for ML training and evaluation.
- Permits use of resulting model weights in the intended public application.
- States whether images or crops may be redistributed.
- Identifies required attribution and deletion procedures.
- Supplies a stable listing or vehicle identifier so duplicate views stay grouped.
- Does not include vendor stock imagery outside the granted rights.

Prefer a dealer-provided export, storage bucket, or documented API. Do not crawl the public dealership website unless the agreement explicitly authorizes automated collection.

## Target taxonomy

Each class represents a visually meaningful US-market generation, not an unsupported exact year:

`corporate_group -> brand -> model -> generation -> body_style`

Required fields:

- Corporate group
- Consumer-facing brand
- Canonical model
- Generation identifier and year range
- US market flag
- Body style
- Optional trim only when it causes a consistent visible difference
- Source citations and review status

Model-year labels remain metadata. Adjacent years with the same body are grouped into one generation. Facelifts may become sub-generations only when the benchmark shows a stable visual distinction.

## 2000-2010 priority brands

- General Motors: Buick, Cadillac, Chevrolet, GMC, Oldsmobile, Pontiac, Saturn, Hummer
- Ford Motor Company: Ford, Lincoln, Mercury
- Toyota Motor Company: Toyota, Lexus, Scion
- American Honda: Honda, Acura
- Chrysler Group: Chrysler, Dodge, Jeep, Plymouth, Ram where applicable
- Nissan North America: Nissan, Infiniti
- Hyundai and Kia
- BMW and MINI
- Subaru, Volkswagen, Mercedes-Benz, Mazda, Audi, Mitsubishi, Volvo
- Jaguar, Land Rover, Porsche, Suzuki, Saab

## Image targets

Use generation-level targets rather than promising equal exact-year coverage:

- High-volume generation: 500-1,000 accepted images.
- Medium-volume generation: 250-500 accepted images.
- Rare generation: 100-250 accepted images plus metric-learning references.
- Benchmark: at least 20 independent vehicles per generation when practical.

No single viewpoint should dominate. Include front, rear, side, front three-quarter, and rear three-quarter views; factory and mildly modified examples; multiple colors; daylight, night, indoor, rain, partial occlusion, and phone-camera compression.

Multiple photographs from one listing are one vehicle group, not independent examples. Split by vehicle/listing/VIN hash so the same physical vehicle cannot cross training, validation, and test sets.

## Intake workflow

1. Register the source and terms in `source-registry.json`.
2. Record a frozen copy or hash of the applicable agreement outside the repository.
3. Import source labels into the normalized CSV contract.
4. Reject non-US, out-of-range, unsupported, or rights-unknown records.
5. Strip EXIF and personal metadata.
6. Detect and redact faces and license plates before human review or training.
7. Compute cryptographic, perceptual, and embedding hashes for deduplication.
8. Group all views of one physical vehicle.
9. Review labels against the taxonomy and generation sources.
10. Freeze benchmark groups before training.
11. Produce a dataset card listing sources, counts, exclusions, known bias, and allowed use.

## Quality controls

- Reject stock renders when the record claims to show a photographed used vehicle.
- Detect watermarks and dealership overlays; retain only when permitted and ensure they do not become label shortcuts.
- Balance recurring dealer backgrounds across brands and splits.
- Audit label noise, especially automatically inferred exact years.
- Track source distribution so one dataset does not dominate a class.
- Keep benchmark images independent from training websites and listing sequences.
- Report accuracy by brand, generation, decade, viewpoint, source, and image quality.

## Training sequence

1. Train and benchmark a brand classifier.
2. Train brand-specific model/generation heads.
3. Add viewpoint prediction.
4. Train whole-car metric embeddings.
5. Train visible-part localization and the part-aware fusion head.
6. Add verified-reference retrieval.
7. Calibrate confidence and unknown rejection on held-out data.
8. Promote a model only when it beats the baseline and satisfies the source registry's deployment rules.

## Completion criteria

The dataset strategy is operational when:

- Every source has a machine-readable rights decision.
- The 2000-2010 generation catalog is reviewed and frozen.
- Benchmark groups are labeled and isolated.
- Dataset imports pass provenance, privacy, deduplication, and taxonomy validation.
- Research-only and production artifacts are physically separated.
- The model card reports per-brand and per-generation metrics and known gaps.
