from __future__ import annotations
"""API Flask para inferencia de clasificacion de razas.

Endpoints:
- GET /health
- POST /predict (multipart/form-data, campo `file`)

Uso rapido:
1) python app.py
2) En otra terminal: curl -X POST -F "file=@imagen.jpg" http://localhost:5000/predict
"""

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image

ARTIFACTS_DIR = Path("artifacts")
MODEL = tf.keras.models.load_model(ARTIFACTS_DIR / "best_model.keras")
META = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
CLASS_NAMES = META["class_names"]
INPUT_SIZE = tuple(META.get("input_size", [224, 224]))

app = Flask(__name__)


def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """Preprocesa bytes de imagen al formato de entrada del modelo."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/health")
def health():
    """Endpoint de salud para verificar carga del servicio/modelo."""
    return jsonify({"status": "ok", "winner_model": META.get("winner_model")})


@app.post("/predict")
def predict():
    """Recibe una imagen y retorna top-3 predicciones."""
    if "file" not in request.files:
        return jsonify({"error": "Debe enviar archivo en campo 'file'"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Nombre de archivo vacio"}), 400

    x = preprocess_bytes(file.read())
    probs = MODEL.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:3]
    preds = [{"class": CLASS_NAMES[i], "probability": float(probs[i])} for i in top_idx]
    return jsonify({"predictions": preds})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
