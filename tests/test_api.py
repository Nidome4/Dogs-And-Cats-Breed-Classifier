"""
Tests for the Animal Classifier API endpoints.
"""

import io


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model"] == "EfficientNet-B0"


class TestIndexEndpoint:
    """Tests for the / endpoint."""

    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Animal Classifier" in response.text


class TestPredictFileEndpoint:
    """Tests for the POST /predict/file endpoint."""

    def test_predict_file_with_valid_image(self, client, sample_image_bytes):
        response = client.post(
            "/predict/file",
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        # The image may or may not contain an animal, so we accept 200 or 400
        assert response.status_code in (200, 400)
        if response.status_code == 200:
            data = response.json()
            assert "animal" in data
            assert "confidence" in data
            assert "inference_time_ms" in data
            assert "top_predictions" in data
            assert isinstance(data["top_predictions"], list)

    def test_predict_file_rejects_invalid_extension(self, client):
        fake_file = io.BytesIO(b"fake content")
        response = client.post(
            "/predict/file",
            files={"file": ("test.txt", fake_file, "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_predict_file_rejects_no_file(self, client):
        response = client.post("/predict/file")
        assert response.status_code == 422


class TestPredictUrlEndpoint:
    """Tests for the POST /predict/url endpoint."""

    def test_predict_url_rejects_invalid_url(self, client):
        response = client.post(
            "/predict/url",
            json={"url": "not-a-valid-url"},
        )
        assert response.status_code == 400

    def test_predict_url_rejects_empty_url(self, client):
        response = client.post(
            "/predict/url",
            json={"url": ""},
        )
        assert response.status_code == 400

    def test_predict_url_rejects_missing_body(self, client):
        response = client.post("/predict/url")
        assert response.status_code == 422
