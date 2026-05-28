from __future__ import annotations

import io

import requests
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_DOWNLOAD_SIZE = 10 * 1024 * 1024
DOWNLOAD_TIMEOUT = 12


def _validate_extension(filename: str) -> None:
    import os

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")


async def load_image_from_upload(upload: UploadFile) -> Image.Image:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    _validate_extension(upload.filename)
    try:
        image = Image.open(io.BytesIO(await upload.read()))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")


def load_image_from_url(url: str) -> Image.Image:
    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        content = response.content
        if len(content) > MAX_DOWNLOAD_SIZE:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

        image = Image.open(io.BytesIO(content))
        image.load()
        return image.convert("RGB")
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {exc}")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="URL content is not a valid image")
