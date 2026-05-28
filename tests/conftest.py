"""
Pytest fixtures for the Animal Classifier test suite.
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image_bytes():
    """Generate a small valid RGB image as bytes (JPEG format)."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_image_file(sample_image_bytes):
    """Return a tuple (filename, bytes, content_type) for upload testing."""
    return ("test_image.jpg", sample_image_bytes, "image/jpeg")


@pytest.fixture
def invalid_file_bytes():
    """Generate invalid (non-image) file bytes."""
    return b"This is not an image file content at all"
