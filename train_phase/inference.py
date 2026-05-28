from __future__ import annotations
"""Script de inferencia offline usando el modelo serializado en `artifacts/`.

Pensado para pruebas rapidas desde terminal sin levantar servidor web.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


class Predictor:
    """Encapsula carga de artefactos, preprocesamiento y prediccion top-k.

    Artefactos esperados:
    - best_model.keras
    - metadata.json
    """

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.model = tf.keras.models.load_model(artifacts_dir / "best_model.keras")
        meta = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
        self.class_names = meta["class_names"]
        self.input_size = tuple(meta.get("input_size", [224, 224]))

    def preprocess(self, image_path: Path) -> np.ndarray:
        """Convierte una imagen a tensor [1, H, W, 3] normalizado en [0,1]."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.input_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image_path: Path, top_k: int = 3):
        """Retorna las `top_k` clases con mayor probabilidad."""
        x = self.preprocess(image_path)
        probs = self.model.predict(x, verbose=0)[0]
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [
            {"class": self.class_names[i], "probability": float(probs[i])}
            for i in top_idx
        ]


def main() -> None:
    """Punto de entrada CLI para inferencia de una sola imagen."""
    parser = argparse.ArgumentParser(description="Inferencia desde modelo guardado.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    p = Predictor(args.artifacts)
    preds = p.predict(args.image, top_k=args.top_k)
    print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
