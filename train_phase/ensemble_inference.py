from __future__ import annotations
"""Inferencia con ensemble de modelos (promedio de probabilidades).

Requiere modelos guardados por `train_models.py` en:
- artifacts/ensemble_models/*.keras
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess(image_path: Path, input_size=(224, 224)) -> np.ndarray:
    """Convierte imagen a tensor batch [1,H,W,3] en rango [0,1]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(input_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def load_ensemble(artifacts_dir: Path):
    """Carga metadata y todos los modelos del ensemble."""
    meta_path = artifacts_dir / "ensemble_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    class_names = meta["class_names"]
    model_files = meta["model_files"]

    models = []
    for mf in model_files:
        model_path = artifacts_dir / "ensemble_models" / mf
        models.append(tf.keras.models.load_model(model_path))

    return models, class_names


def predict_ensemble(models, x: np.ndarray) -> np.ndarray:
    """Promedia probabilidades de cada modelo."""
    probs = [m.predict(x, verbose=0)[0] for m in models]
    return np.mean(np.stack(probs, axis=0), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia por ensemble de modelos.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    models, class_names = load_ensemble(args.artifacts)
    x = preprocess(args.image)
    probs = predict_ensemble(models, x)

    top_idx = np.argsort(probs)[::-1][: args.top_k]
    preds = [{"class": class_names[i], "probability": float(probs[i])} for i in top_idx]
    print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
