import os
from pathlib import Path

ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_DETECTION_THRESHOLD = 0.55
CROP_PADDING_RATIO = 0.05
CLASSIFIER_MODEL_ID = "therealcyberlord/stanford-car-vit-patch16"
LOW_CLASSIFICATION_CONFIDENCE = 0.65
UNKNOWN_CLASSIFICATION_CONFIDENCE = 0.35
UNKNOWN_CLASSIFICATION_MARGIN = 0.08
REFERENCE_SUPPORT_THRESHOLD = 0.80
TOP_PREDICTIONS = 5

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
ALLOWED_ORIGINS = tuple(
    origin.strip().rstrip("/")
    for origin in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)).split(",")
    if origin.strip()
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_GALLERY_PATH = PROJECT_ROOT / "model-files" / "reference-gallery.jsonl"
GENERATION_CATALOG_PATH = PROJECT_ROOT / "model-files" / "vehicle-generations.json"
CONTRIBUTIONS_ROOT = PROJECT_ROOT / "backend" / "data" / "contributions"
FEEDBACK_DATABASE_PATH = PROJECT_ROOT / "backend" / "data" / "feedback.sqlite3"
