from pathlib import Path

import pytest
from PIL import Image

from backend.app.utils.images import ImageValidationError, load_validated_image


@pytest.mark.parametrize("extension,image_format", [(".jpg", "JPEG"), (".png", "PNG"), (".webp", "WEBP")])
def test_loads_supported_images(
    tmp_path: Path, extension: str, image_format: str
) -> None:
    path = tmp_path / f"vehicle{extension}"
    Image.new("RGB", (32, 24), "blue").save(path, format=image_format)

    loaded = load_validated_image(path)

    assert loaded.mode == "RGB"
    assert loaded.size == (32, 24)


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "vehicle.gif"
    Image.new("RGB", (10, 10)).save(path, format="GIF")

    with pytest.raises(ImageValidationError, match="Unsupported file extension"):
        load_validated_image(path)


def test_rejects_fake_image(tmp_path: Path) -> None:
    path = tmp_path / "vehicle.jpg"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ImageValidationError, match="not a valid"):
        load_validated_image(path)


def test_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "vehicle.png"
    Image.new("RGB", (10, 10)).save(path, format="PNG")

    with pytest.raises(ImageValidationError, match="too large"):
        load_validated_image(path, max_bytes=1)

