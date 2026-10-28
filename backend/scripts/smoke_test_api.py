import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app

CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real-model request through the FastAPI application."
    )
    parser.add_argument("--input", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    content_type = CONTENT_TYPES.get(args.input.suffix.lower())
    if content_type is None:
        raise SystemExit("Error: input must be JPG, JPEG, PNG, or WEBP.")
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": (args.input.name, args.input.read_bytes(), content_type)},
        )
    payload = response.json()
    if response.status_code != 200:
        print(json.dumps(payload, indent=2))
        return 1
    payload["attention_image_data_url"] = "<base64 JPEG omitted>"
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
