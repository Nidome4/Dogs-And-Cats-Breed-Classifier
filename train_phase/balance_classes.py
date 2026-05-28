from __future__ import annotations
"""Balanceo de clases por data augmentation.

Este script iguala el numero de imagenes por clase dentro de un directorio con
estructura:
    input_dir/
      clase_1/*.jpg
      clase_2/*.jpg
      ...

Estrategia:
- Copia todas las imagenes originales al directorio de salida.
- Detecta la clase con mas muestras (o usa --target-count si se define).
- Para clases con menos muestras, genera imagenes nuevas con augmentations
  aleatorias hasta alcanzar el target.

Nota importante para novatos:
- Este script se recomienda para `train/` solamente.
- No balances `val/` ni `test/` para no distorsionar la evaluacion real.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def load_rgb_image(path: Path) -> np.ndarray:
    """Carga una imagen y la retorna como arreglo uint8 RGB."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb_image(arr: np.ndarray, path: Path) -> None:
    """Guarda un arreglo RGB uint8 como JPG."""
    Image.fromarray(arr).save(path, format="JPEG", quality=95)


def augment_image(img_uint8: np.ndarray, rng: random.Random) -> np.ndarray:
    """Aplica augmentations aleatorias y retorna imagen uint8.

    Esto funciona como data generator "offline": crea nuevos archivos en disco.
    """
    x = tf.convert_to_tensor(img_uint8, dtype=tf.float32) / 255.0

    # Flips aleatorios
    if rng.random() < 0.5:
        x = tf.image.flip_left_right(x)
    if rng.random() < 0.2:
        x = tf.image.flip_up_down(x)

    # Rotaciones multiples de 90 para evitar recortes
    k = rng.randint(0, 3)
    x = tf.image.rot90(x, k=k)

    # Brillo y contraste
    x = tf.image.random_brightness(x, max_delta=0.15)
    x = tf.image.random_contrast(x, lower=0.8, upper=1.2)

    # Saturacion y matiz
    x = tf.image.random_saturation(x, lower=0.8, upper=1.2)
    x = tf.image.random_hue(x, max_delta=0.03)

    # Recorte aleatorio + resize al tamano original
    h, w = img_uint8.shape[0], img_uint8.shape[1]
    crop_scale = rng.uniform(0.85, 1.0)
    crop_h = max(1, int(h * crop_scale))
    crop_w = max(1, int(w * crop_scale))
    x = tf.image.random_crop(x, size=[crop_h, crop_w, 3])
    x = tf.image.resize(x, [h, w], method="bilinear")

    x = tf.clip_by_value(x, 0.0, 1.0)
    out = (x.numpy() * 255.0).astype(np.uint8)
    return out


def get_class_images(class_dir: Path) -> list[Path]:
    """Retorna imagenes validas dentro de una carpeta de clase."""
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def balance_classes(input_dir: Path, output_dir: Path, target_count: int | None, seed: int) -> None:
    """Balancea clases por aumento de datos hasta alcanzar el mismo numero."""
    rng = random.Random(seed)

    class_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No se encontraron carpetas de clase en: {input_dir}")

    class_to_files: dict[str, list[Path]] = {}
    for cdir in class_dirs:
        files = get_class_images(cdir)
        if files:
            class_to_files[cdir.name] = files

    if not class_to_files:
        raise ValueError("No se encontraron imagenes validas para balancear.")

    counts = {cls: len(files) for cls, files in class_to_files.items()}
    max_count = max(counts.values())
    goal = target_count if target_count is not None else max_count

    if goal < max_count:
        raise ValueError(
            f"target_count={goal} es menor que la clase maxima ({max_count}). "
            "Use un valor >= max_count o deje el default."
        )

    print("Conteo original por clase:")
    for cls, n in sorted(counts.items()):
        print(f"- {cls}: {n}")
    print(f"\nObjetivo por clase: {goal}\n")

    for cls, files in sorted(class_to_files.items()):
        out_cls = output_dir / cls
        out_cls.mkdir(parents=True, exist_ok=True)

        # Copiar originales
        for i, src in enumerate(files):
            dst = out_cls / f"{cls}_orig_{i:05d}.jpg"
            arr = load_rgb_image(src)
            save_rgb_image(arr, dst)

        current = len(files)
        needed = goal - current
        if needed <= 0:
            print(f"{cls}: ya cumple ({current}/{goal})")
            continue

        # Generar augmentations hasta alcanzar objetivo
        for j in range(needed):
            base_path = files[rng.randrange(0, len(files))]
            base_arr = load_rgb_image(base_path)
            aug_arr = augment_image(base_arr, rng)
            dst = out_cls / f"{cls}_aug_{j:05d}.jpg"
            save_rgb_image(aug_arr, dst)

        print(f"{cls}: {current} -> {goal} (generadas {needed})")

    print("\nBalanceo completado.")


def main() -> None:
    """CLI principal."""
    parser = argparse.ArgumentParser(description="Balancea clases con data augmentation.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Carpeta con subcarpetas por clase.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Carpeta de salida balanceada.")
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Objetivo por clase. Si se omite, usa el maximo de clases existentes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"No existe input-dir: {args.input_dir}")

    balance_classes(args.input_dir, args.output_dir, args.target_count, args.seed)


if __name__ == "__main__":
    main()
