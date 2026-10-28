import io
import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from backend.app.core.config import (
ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_FORMATS,
    MAX_UPLOAD_BYTES,
)

ALLOWED_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/webp"}
)


class ImageValidationError(ValueError):
    """Raised when an input cannot safely be treated as a supported image."""


def load_validated_image(path: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> Image.Image:
    path = path.resolve()
    if not path.is_file():
        raise ImageValidationError(f"Image file does not exist: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ImageValidationError(f"Unsupported file extension. Allowed: {allowed}")

    size = path.stat().st_size
    if size == 0:
        raise ImageValidationError("The image file is empty.")
    if size > max_bytes:
        raise ImageValidationError(
            f"Image is too large ({size / 1024 / 1024:.1f} MB). "
            f"Maximum size is {max_bytes / 1024 / 1024:.0f} MB."
        )

    return _decode_validated_image(path.read_bytes())


def load_validated_image_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Image.Image:
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ImageValidationError(f"Unsupported file extension. Allowed: {allowed}")
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError("Unsupported content type. Upload a JPG, PNG, or WEBP image.")
    if not data:
        raise ImageValidationError("The image file is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"Image is too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum size is {max_bytes / 1024 / 1024:.0f} MB."
        )
    return _decode_validated_image(data)


def _decode_validated_image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as candidate:
                detected_format = candidate.format
                candidate.verify()
            if detected_format not in ALLOWED_IMAGE_FORMATS:
                raise ImageValidationError(
                    "The file contents are not a supported JPG, PNG, or WEBP image."
                )
            with Image.open(io.BytesIO(data)) as candidate:
                return candidate.convert("RGB")
    except Image.DecompressionBombError as error:
        raise ImageValidationError("The image dimensions are too large to process safely.") from error
    except Image.DecompressionBombWarning as error:
        raise ImageValidationError("The image dimensions are too large to process safely.") from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ImageValidationError("The file is not a valid, readable image.") from error
