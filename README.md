# AI Car Identifier

A free, local-first vehicle identification application. The finished pipeline will detect a vehicle, crop it, classify its make/model/generation, and return five ranked predictions with confidence scores.

> Accuracy is not guaranteed. Camera angle, lighting, occlusion, damage, modifications, and visually similar vehicles can all affect results. When evidence does not support an exact year, the application will return a generation or year range.

## Recommended free architecture

| Layer | First-version choice | Why |
|---|---|---|
| Detection | TorchVision SSDLite320 MobileNetV3, COCO pretrained | Free, lightweight, automatically downloaded, and detects car, truck, bus, and motorcycle classes |
| Classification | Stanford Cars ViT baseline plus verified embedding-reference retrieval | Immediately runnable; reference corroboration and a replaceable interface prevent unsupported softmax scores from being treated as proof |
| AI/API backend | Python 3.11, PyTorch, FastAPI | Meets the project requirements and has strong ML support |
| Frontend | Next.js with TypeScript | Simple upload UI, image preview, and production build |
| Storage | SQLite/local files for development; replaceable object storage later | Images are retained only after separate training consent and are stripped of EXIF metadata |
| Training | Google Colab or Kaggle | Free GPU tiers support incremental fine-tuning, although full 1980-2026 coverage is a substantial dataset project |
| Deployment | Hugging Face Spaces or Render for backend; Vercel for frontend | Free tiers may change and can cold-start, so verify limits before deployment |

The detector and classifier will live behind separate interfaces. That lets either model be replaced without changing upload validation, API response shapes, or the frontend.

## Complete planned folder structure

```text
AI_Car/
|-- .gitignore
|-- README.md
|-- backend/
|   |-- requirements.txt
|   |-- requirements-dev.txt
|   |-- app/
|   |   |-- __init__.py
|   |   |-- main.py                    # Phase 3 FastAPI application
|   |   |-- api/routes/predict.py      # Phase 3 upload endpoint
|   |   |-- core/config.py             # Shared limits and model settings
|   |   |-- schemas/prediction.py      # API response types
|   |   |-- services/detector.py       # Phase 1 vehicle detection/cropping
|   |   |-- services/classifier.py     # Phase 2 classifier and attention evidence
|   |   |-- services/reference_store.py # Verified reference-image retrieval
|   |   |-- services/feedback.py       # Consent-gated confirmation storage
|   |   `-- utils/images.py            # Image decode and validation
|   |-- data/
|   |   |-- input/                     # Local test images (not committed)
|   |   |-- output/                    # Generated crops and attention maps
|   |   `-- contributions/             # Consented images pending review
|   |-- scripts/
|   |   |-- phase1_detect.py           # Phase 1 command-line prototype
|   |   `-- phase2_identify.py         # Phase 2 prediction/confirmation CLI
|   `-- tests/
|       `-- test_images.py
|-- frontend/                           # Phase 4 Next.js application
|-- model-files/                        # Weights, labels, and reference gallery
`-- training/
    |-- add_reference.py                # Curator-only verified reference ingestion
    |-- sync_nhtsa_catalog.py           # U.S. make/model/year catalog sync
    `-- data/                            # Catalog inputs and generated metadata
```

Phases 1-5 are complete for local development. Phase 6 selection, rejection, taxonomy, benchmark, and dataset-normalization infrastructure is implemented below; representative benchmark images and licensed training archives are still required before final accuracy training.

## Phase 1 - detect and crop one vehicle

### Objective

Accept one local JPG, JPEG, PNG, or WEBP image, validate it, find the highest-confidence road vehicle, add a small amount of padding, and save the crop. If no qualifying vehicle is found, show a clear error and do not create a misleading result.

### Required files

- `backend/requirements.txt`
- `backend/app/core/config.py`
- `backend/app/utils/images.py`
- `backend/app/services/detector.py`
- `backend/scripts/phase1_detect.py`

### Windows PowerShell installation

Run each command from the project root, one at a time.

1. Create an isolated Python environment:

   ```powershell
   py -3.11 -m venv .venv
   ```

2. Allow activation scripts for this PowerShell process only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

3. Activate the environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Upgrade the Python package installer:

   ```powershell
   python -m pip install --upgrade pip
   ```

5. Install Phase 1 runtime packages:

   ```powershell
   python -m pip install -r .\backend\requirements.txt
   ```

TorchVision downloads the pretrained detector weights on the first run and caches them in the current user's PyTorch cache. An internet connection is required once.

### Run

1. Put one test image in `backend\data\input`, for example `car.jpg`.

2. Run the prototype:

   ```powershell
   python -m backend.scripts.phase1_detect --input .\backend\data\input\car.jpg --output .\backend\data\output\car-crop.jpg
   ```

Optional: lower or raise the detection threshold (default `0.55`):

```powershell
python -m backend.scripts.phase1_detect --input .\backend\data\input\car.jpg --output .\backend\data\output\car-crop.jpg --threshold 0.65
```

### Testing

Install development test packages:

```powershell
python -m pip install -r .\backend\requirements-dev.txt
```

Run the fast validation tests (these do not download model weights):

```powershell
python -m pytest .\backend\tests
```

Manual test cases:

1. A clear car photo should save a crop and print its class and confidence.
2. A scenery or indoor photo should print `No vehicle detected` and exit without a crop.
3. A renamed text file or broken image should be rejected as invalid.
4. A file larger than 10 MB should be rejected before decoding.
5. A valid `.webp` file should be accepted.

### Expected result

Example console output:

```text
Detected car with 91.4% confidence.
Crop saved to: C:\...\backend\data\output\car-crop.jpg
Bounding box: (104, 88, 923, 611)
```

The output image contains the best detected vehicle with 5% padding around its bounding box.

### Common errors and fixes

- `Requested Python version is not installed`: install 64-bit Python 3.11 from python.org, then reopen PowerShell.
- PowerShell blocks `Activate.ps1`: run the process-scoped execution-policy command above; it does not permanently change the machine policy.
- Model download fails: confirm internet access and rerun. After a successful first download, subsequent runs use the cache.
- `No vehicle detected`: use a clearer image or try `--threshold 0.40`. Very small, heavily occluded, unusual, or non-road vehicles may not be recognized by the COCO detector.
- Install runs out of disk space: PyTorch is a large dependency; allow roughly 2 GB for the environment and cached weights.
- Crop is the wrong vehicle: Phase 1 intentionally chooses the highest-confidence vehicle. A later UI could offer selection, but that is outside the simple first version.

## Phase 2 - make, model, and year-range classification

### Objective

Pass the Phase 1 vehicle crop to a fine-grained car classifier and return the five most likely make/model matches with confidence percentages. The baseline uses the Apache-2.0 `therealcyberlord/stanford-car-vit-patch16` model, a ViT fine-tuned on the 196 Stanford Cars classes. Its model card reports about 86% test accuracy, but the dataset is small and does not cover newer vehicles.

The confidence is the model's softmax score, not a guarantee or a calibrated probability of correctness. Predictions below 65% are clearly marked low-confidence. For those results, the dataset's reference year is widened to an estimated five-year range rather than presented as an exact year.

### Required files

- `backend/app/schemas/prediction.py`
- `backend/app/services/classifier.py`
- `backend/app/services/reference_store.py`
- `backend/app/services/feedback.py`
- `backend/scripts/phase2_identify.py`
- `backend/tests/test_classifier.py`
- `backend/tests/test_reference_store.py`
- `backend/tests/test_feedback.py`
- `training/add_reference.py`
- `training/sync_nhtsa_catalog.py`
- Updated `backend/requirements.txt`

### Windows PowerShell installation

If a new PowerShell window was opened, move to the project and activate the existing environment:

```powershell
cd "path\to\AI_Car"
```

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the newly added Phase 2 dependency:

```powershell
python -m pip install -r .\backend\requirements-dev.txt
```

This installs Hugging Face Transformers. The classifier weights download automatically on the first Phase 2 run and are then reused from the local Hugging Face cache.

### Run

Use the same test image from Phase 1:

```powershell
python -m backend.scripts.phase2_identify --input .\backend\data\input\car.jpg --crop-output .\backend\data\output\phase2-car-crop.jpg --attention-output .\backend\data\output\phase2-attention.jpg
```

The command runs the complete local AI pipeline: validate image, detect vehicle, crop, preprocess, classify, and print five ranked matches as JSON. The optional attention image indicates where the transformer focused; it is supporting evidence, not proof of identity.

To record that the second candidate is correct without retaining either image:

```powershell
python -m backend.scripts.phase2_identify --input .\backend\data\input\car.jpg --confirm 2
```

To explicitly consent to storing metadata-free copies of the original and crop for later review and training:

```powershell
python -m backend.scripts.phase2_identify --input .\backend\data\input\car.jpg --confirm 2 --user-confidence certain --contribute
```

Confirmation and image consent are separate. Without `--contribute`, no uploaded pixels are retained by the feedback service. Contributed files are re-encoded without EXIF data under `backend/data/contributions`, ignored by Git, and marked `pending` in `backend/data/feedback.sqlite3`. They must be reviewed before training.

### Testing

Run all Phase 1 and Phase 2 unit tests:

```powershell
python -m pytest .\backend\tests
```

Manual test cases:

1. Run a clear photo of a Stanford Cars-era vehicle and confirm five predictions are returned in descending confidence order.
2. Run a recent vehicle and confirm low confidence is presented honestly; the dataset does not cover recent models.
3. Run a visually similar trim or modified vehicle and inspect all five matches rather than trusting only the first.
4. Run a non-vehicle image and confirm classification never runs because detection returns a clear error first.

### Expected result

The output is shaped like this:

```json
{
  "detection": {
    "vehicle_type": "car",
    "confidence_percent": 94.0,
    "bounding_box": [104, 88, 923, 611]
  },
  "classification": {
    "predictions": [
      {
        "make": "Audi",
        "model": "S5 Coupe",
        "year_range": "2010-2014 (estimated range)",
        "confidence_percent": 42.0,
        "raw_label": "Audi S5 Coupe 2012"
      }
    ],
    "low_confidence": true,
    "message": "Low-confidence result. Treat all matches and year ranges as possibilities.",
    "model_id": "therealcyberlord/stanford-car-vit-patch16",
    "limitations": "Stanford Cars has limited, mostly older reference classes. A high model score does not prove that the real-world make, model, or year is correct."
  }
}
```

The real response contains exactly five prediction objects. Detection and classification confidence values are returned as percentages. Each prediction also contains a `references` array. It remains empty until verified gallery records have been added; the application never invents citations for hidden training images.

### Common errors and fixes

- First run appears paused: the ViT classifier has roughly 86 million parameters and must download once. Leave the command running and check the internet connection.
- `No module named transformers`: activate `.venv`, then rerun the Phase 2 installation command.
- Hugging Face download or certificate error: retry on a stable connection. A proxy or antivirus HTTPS inspection may require local certificate configuration.
- Results describe an older but similar car: Stanford Cars labels mainly represent vehicles available around 2012 or earlier. Treat this model as a functional baseline and later fine-tune on a dataset whose license permits the intended deployment. CompCars is restricted to non-commercial research and is not the default production training source.
- Confidence is low: inspect all five matches. Angle, lighting, occlusion, damage, aftermarket parts, and similar designs can reduce confidence.
- CPU inference is slow: this ViT is substantially larger than the detector. A future classifier can replace it without changing the detector or result schema.

### Adding verified reference evidence

Only add images whose identity and reuse license have been checked. This curator command detects the vehicle, creates its ViT embedding, and appends provenance to the local reference gallery:

```powershell
python -m training.add_reference --input .\licensed-images\forester.jpg --make Subaru --model Forester --generation "Second generation" --year-start 2003 --year-end 2008 --source-url "https://source.example/image" --license "CC BY 4.0" --attribution "Photographer name"
```

Restart the prediction command after adding records so the classifier reloads the gallery. It will attach the three closest same-make references to each candidate. JSONL is intentionally simple for the prototype; use FAISS or a vector database once the gallery contains tens of thousands of images.

### Building the 1980-2026 U.S. taxonomy

NHTSA vPIC provides U.S.-sale/import make and model metadata and supports make/model/year lookup after 1995. Synchronize the starter list for 1996-2026 with:

```powershell
python -m training.sync_nhtsa_catalog --makes-file .\training\data\us_makes.txt --start-year 1996 --end-year 2026 --output .\training\data\nhtsa-us-catalog.json
```

This command performs many rate-limited requests and can take time. Expand `us_makes.txt` as historical and niche makes are reviewed. Years 1980-1995 must be curated into `training/data/legacy_us_catalog.csv` from authoritative sources because this vPIC operation does not cover them.

Catalog metadata is not image training data. Broad recognition still requires licensed, verified images for each generation, balanced across angles and conditions. Phase 2 supplies the ingestion, retrieval, feedback, and evidence mechanisms; it does not claim coverage for a generation until suitable references/training examples actually exist.

Validate the curated generation file after every edit:

```powershell
python -m training.validate_generation_catalog
```

Only `verified` records with non-empty source URLs are used during inference. Pending records remain visible to curators but cannot silently influence predictions. The included catalog is intentionally a small seed, not a claim of complete 1980-2026 coverage.

## Phase 3 - FastAPI service

### Objective

Expose the local detector, classifier, evidence, references, and consent-aware feedback workflow through a stable HTTP API. Models load once when the server starts and are reused across requests.

The API does not persist uploaded image pixels during prediction. It retains a short-lived prediction result and SHA-256 fingerprint for up to 30 minutes. The attention image is returned directly to the client. If training consent is later granted, the client must resubmit the same image and the server verifies its fingerprint before storing metadata-free copies.

### Required files

- `backend/app/main.py`
- `backend/app/api/routes/predict.py`
- `backend/app/schemas/api.py`
- `backend/app/services/inference.py`
- `backend/app/services/prediction_cache.py`
- Updated `backend/app/utils/images.py`
- `backend/scripts/smoke_test_api.py`
- `backend/tests/test_api.py`
- `backend/tests/test_generation_catalog.py`

### Installation

Activate the existing environment, then install the updated requirements:

```powershell
cd "path\to\AI_Car"
```

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install -r .\backend\requirements-dev.txt
```

### Run

Start FastAPI on the local machine:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Model loading can take several seconds. Leave this PowerShell window open. Open the interactive API documentation in a browser at `http://127.0.0.1:8000/docs`.

In a second PowerShell window, submit an image:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/predict" -F "file=@backend/data/input/car.jpg;type=image/jpeg"
```

Use the returned `prediction_id` to confirm candidate 1 without retaining the image:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/feedback" -F "prediction_id=REPLACE_WITH_PREDICTION_ID" -F "selected_rank=1" -F "user_confidence=certain" -F "consent_to_training=false"
```

To confirm and explicitly contribute the matching image:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/feedback" -F "prediction_id=REPLACE_WITH_PREDICTION_ID" -F "selected_rank=1" -F "user_confidence=certain" -F "consent_to_training=true" -F "file=@backend/data/input/car.jpg;type=image/jpeg"
```

### API routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/v1/predict` | Validate one image and return detection, five candidates, evidence, uncertainty, and attention image |
| `POST` | `/api/v1/feedback` | Confirm a ranked candidate, with separate optional training consent |
| `GET` | `/api/v1/references/{reference_id}` | Return the verified source, license, attribution, and vehicle metadata for a reference |
| `GET` | `/api/v1/health` | Report loaded model ID and reference/generation counts |

### Testing

Run the complete backend suite:

```powershell
python -m pytest .\backend\tests
```

Run one request through the real models and FastAPI stack without starting a persistent server:

```powershell
python -m backend.scripts.smoke_test_api --input .\backend\data\input\car.jpg
```

### Expected result

`POST /api/v1/predict` returns HTTP 200 with a unique prediction ID, detection metadata, five classification candidates, an uncertainty message, reference support, generation fields, and a base64 JPEG attention image. Invalid or oversized images return HTTP 400. A valid image with no detected vehicle returns HTTP 422.

Feedback without consent records the selected label but stores no image. Consented feedback requires the original file and rejects a different file with HTTP 409. Prediction metadata expires after 30 minutes or when the bounded in-memory cache evicts it.

### Common errors and fixes

- Server seems paused at startup: wait for both pretrained models to load; the first run may also download weights.
- `Address already in use`: stop the previous Uvicorn process with `Ctrl+C` or choose another port such as `--port 8001`.
- HTTP 404 on feedback: the server restarted, the prediction expired, or an incorrect prediction ID was supplied. Predict again.
- HTTP 409 on consented feedback: resubmit the exact original file used for that prediction.
- Browser CORS error: Phase 3 permits `localhost:3000` and `127.0.0.1:3000`. Add the deployed frontend origin through configuration before deployment.
- Large JSON response: the attention visualization is currently embedded as a data URL to avoid retaining it server-side. Phase 5 may move it to short-lived object storage with an explicit retention policy.

## Deployment

Deploy the FastAPI backend to an ML-capable host such as Hugging Face Spaces or Render. Set its `ALLOWED_ORIGINS` environment variable to the exact Vercel frontend origin, without a trailing slash. Multiple origins can be separated with commas.

Import this repository in Vercel and configure:

- Root Directory: `frontend`
- Framework Preset: Next.js
- Environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.example`

Vercel will install the frontend dependencies and run `npm run build`. The PyTorch backend is intentionally deployed separately because its model size, memory use, startup time, and persistent-data requirements are not a good fit for the frontend's serverless deployment.

## Next phases

## Phase 4 - Next.js upload and results interface

### Objective

Provide a responsive browser interface for uploading one vehicle photo, previewing it, calling FastAPI, displaying five ranked matches and attention evidence, confirming the correct candidate, and optionally contributing the image for review.

The privacy popup is informational and disappears when the user clicks anywhere, selects its close button, or presses Escape. Dismissing it is not treated as consent. Persistent storage only occurs after the separate `Contribute this image` button is selected.

### Required files

- `frontend/app/page.tsx`
- `frontend/app/layout.tsx`
- `frontend/app/globals.css`
- `frontend/.env.example`
- `frontend/tests/rendered-html.test.mjs`
- `frontend/package.json`
- `frontend/README.md`

### Installation and run commands

Start FastAPI from the repository root in the first PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window, move to the frontend:

```powershell
cd "path\to\AI_Car\frontend"
```

```powershell
npm install
```

```powershell
npm run dev
```

Open `http://localhost:3000`. The default API URL is `http://127.0.0.1:8000`. For another backend, copy `frontend/.env.example` to `frontend/.env.local` and change `NEXT_PUBLIC_API_URL`.

### Testing

```powershell
cd "path\to\AI_Car\frontend"
```

```powershell
npm test
```

Manual tests:

1. Dismiss the privacy popup by clicking the backdrop, popup, close button, and Escape on separate reloads.
2. Upload one file of each supported format and verify its preview.
3. Try an unsupported or over-10-MB file and verify the local error message.
4. Stop FastAPI and verify the service-unavailable message.
5. Analyze a vehicle and confirm that five candidates and the attention image appear.
6. Confirm a candidate without contributing and verify that the status says the image was not stored.
7. Contribute after confirmation and verify the sanitized-copy review message.
8. Resize to a narrow viewport and verify that cards, results, and controls stack cleanly.
9. Upload a photo with several vehicles, select each numbered outline, and verify the bright scan area follows the selection.
10. Use **Snap outline** to draw a rough rectangle and verify that it snaps to the closest detected vehicle.
11. Use **Manual brush**, change the brush size, erase part of the mask, and verify only the painted vehicle is analyzed.

### Expected result

The first viewport presents the product purpose and a focused upload card. After analysis, the page scrolls to a primary result, uncertainty state, attention evidence, five selectable candidates, and feedback controls. All AI limitations remain visible and no paid service is used.

### Common errors and fixes

- `npm` is not recognized: install current Node.js LTS, reopen PowerShell, and verify with `node --version`.
- Frontend loads but analysis fails: start FastAPI on port 8000 and confirm `http://127.0.0.1:8000/api/v1/health` returns JSON.
- Port 3000 is already used: stop the previous frontend with `Ctrl+C` before starting another.
- Contribution is rejected as mismatched: select and resubmit the same local image used for the prediction.
- Prediction expired: analyze the image again; Phase 3 retains prediction metadata for 30 minutes.

The browser-to-FastAPI connection originally planned as Phase 5 is complete for local development. Production URLs and hosting configuration remain part of deployment preparation.

## Phase 6 - selection controls and accuracy evaluation

### Objective

Make the target vehicle explicit before classification and establish measurable accuracy before expanding the training data. The interface now offers automatic multi-vehicle outlines, an assisted rectangle that snaps to the nearest detected vehicle, and a manual painted mask with brush-size, eraser, and clear controls.

### Required files

- `frontend/app/VehicleSelector.tsx`
- `frontend/app/page.tsx`
- `frontend/app/globals.css`
- `backend/app/services/detector.py`
- `backend/app/services/inference.py`
- `backend/app/api/routes/predict.py`
- `backend/tests/test_api.py`

### How it works

`POST /api/v1/detect` returns every road-vehicle box above the detector threshold. The browser overlays those boxes at their original-image coordinates. Automatic mode sends the selected box; assisted mode sends the snapped or user-drawn box; manual mode sends both the painted mask and its smallest enclosing box. The backend validates the selection, removes unpainted pixels from manual crops, classifies only the selected crop, and retains no pixels unless the user separately consents after confirming a result.

### Expected result

Background vehicles no longer have to compete for the single automatic crop, and difficult detector cases can be corrected without editing the source image. This improves target selection but does not expand the classifier's 196 Stanford Cars classes.

### Dataset timing

Add VMMRdb, CompCars, Vehicle-1M, and a verified NHTSA-derived taxonomy after the representative evaluation set and label schema are frozen. Importing them earlier would create incompatible year, trim, body-style, and market labels without revealing whether a later accuracy change is an improvement. Dataset licenses and redistribution terms must be reviewed before any images or trained artifacts are published.

The normalized manifest, benchmark scorer, part-annotation contract, and part-aware fusion head are now in `training/`. The official NHTSA sync produced `training/data/nhtsa-us-catalog.json` with 15,487 sourced make/model/year records covering 1996-2026, 45 requested makes with results, and 1,662 unique make/model names. Exact-make filtering prevents vPIC partial matches such as trailer manufacturers containing “Ram” from contaminating the catalog.

The current classifier also has a provisional unknown/ambiguous rejection policy. It abstains when the leading model score is below 35% or the two leading scores are less than eight percentage points apart. These thresholds are deliberately labeled provisional and must be calibrated against the frozen benchmark before deployment.

For the complete PowerShell workflow, dataset CSV contract, leakage rules, benchmark commands, and part-aware model notes, see `training/README.md`.

The acquisition and licensing policy is defined in `training/DATASET_STRATEGY.md`, with enforceable source decisions in `training/data/source-registry.json`. Facebook Marketplace and unlicensed dealership scraping are blocked. Direct dealership partnerships and user submissions are conditional on explicit image rights, ML-training permission, deployment permission, provenance, and privacy cleanup. CompCars and Vehicle-1M are kept research-only under their official terms; VMMRdb remains quarantined until its official image-use terms are obtained and reviewed.

## Next phases

- Phase 6 data gate: add correctly labeled benchmark images, review/download licensed VMMRdb, CompCars, and Vehicle-1M archives, and curate sourced 1980-1995 plus generation metadata.
- Phase 6 training track: run the baseline benchmark, fine-tune the replaceable whole-car and part-aware models, then calibrate confidence and unknown rejection.
- Phase 7: document currently available free deployment paths and their limitations.
