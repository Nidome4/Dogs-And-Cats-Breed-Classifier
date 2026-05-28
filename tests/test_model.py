"""
Tests for the EfficientNet-B0 model wrapper.
"""

from PIL import Image

from app.model import AnimalClassifier, get_classifier


class TestAnimalClassifier:
    """Tests for the AnimalClassifier class."""

    def test_model_loads_successfully(self):
        """Test that the model can be loaded without errors."""
        classifier = get_classifier()
        assert classifier is not None

    def test_prediction_returns_valid_structure(self):
        """Test that predict returns the expected dictionary structure."""
        classifier = get_classifier()
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = classifier.predict(image)

        assert "probabilities" in result
        assert "inference_time_ms" in result
        assert isinstance(result["probabilities"], list)
        assert len(result["probabilities"]) == 1000
        assert isinstance(result["inference_time_ms"], int)
        assert result["inference_time_ms"] >= 0

    def test_probabilities_sum_to_one(self):
        """Test that output probabilities sum approximately to 1."""
        classifier = get_classifier()
        image = Image.new("RGB", (224, 224), color=(200, 100, 50))
        result = classifier.predict(image)
        total = sum(result["probabilities"])
        assert abs(total - 1.0) < 0.01

    def test_handles_rgba_image(self):
        """Test that RGBA images are handled correctly."""
        classifier = get_classifier()
        image = Image.new("RGBA", (300, 300), color=(128, 128, 128, 255))
        result = classifier.predict(image)
        assert len(result["probabilities"]) == 1000

    def test_handles_grayscale_image(self):
        """Test that grayscale images are handled correctly."""
        classifier = get_classifier()
        image = Image.new("L", (256, 256), color=128)
        result = classifier.predict(image)
        assert len(result["probabilities"]) == 1000
