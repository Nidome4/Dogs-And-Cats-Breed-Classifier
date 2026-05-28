from __future__ import annotations
"""Data Generator real (online) para entrenamiento con Keras.

Este archivo implementa generadores por lote usando
`tf.keras.preprocessing.image.ImageDataGenerator`, que NO es lo mismo que:
- data loader simple (solo cargar datos), ni
- data augmentation offline (crear archivos nuevos en disco).

Aqui la augmentacion ocurre EN TIEMPO REAL durante `model.fit(...)`.
"""

import argparse
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_data_generators(
    data_dir: Path,
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """Crea generadores train/val/test desde carpetas.

    Estructura esperada:
        data_dir/
          train/<clase>/*.jpg
          val/<clase>/*.jpg
          test/<clase>/*.jpg
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "No se encontro estructura train/val/test. "
            f"Revisado en: {data_dir}"
        )

    # Generator de entrenamiento: incluye augmentation online.
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Validacion/Test: solo reescalado (sin augmentation).
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
    )

    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def main() -> None:
    """Prueba rapida del generador desde CLI."""
    parser = argparse.ArgumentParser(description="Construye data generators online para train/val/test.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset_701515"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train_gen, val_gen, test_gen = build_data_generators(
        args.data_dir,
        img_size=IMG_SIZE,
        batch_size=args.batch_size,
    )

    print("Generadores creados correctamente")
    print(f"Train samples: {train_gen.samples}")
    print(f"Val samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Num classes: {train_gen.num_classes}")


if __name__ == "__main__":
    main()
