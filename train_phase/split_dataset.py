from __future__ import annotations
"""Genera una particion estratificada 70/15/15 por clase para imagenes JPG/JPEG.

Uso principal:
    python split_dataset.py --source images --target dataset_701515 --mode copy

Para novatos:
- `train` (70%): para aprender.
- `val` (15%): para decidir si el modelo mejora durante entrenamiento.
- `test` (15%): para medir rendimiento final sin sesgo.
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British", "Egyptian", "Maine", "Persian",
    "Ragdoll", "Russian", "Siamese", "Sphynx", "american", "basset", "beagle", "boxer",
    "chihuahua", "english", "german", "great", "havanese", "japanese", "keeshond", "leonberger",
    "miniature", "newfoundland", "pomeranian", "pug", "saint", "samoyed", "scottish", "shiba",
    "staffordshire", "wheaten", "yorkshire",
]
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def detect_class(name: str) -> str | None:
    """Detecta la clase a partir del nombre del archivo.

    Se ordenan las clases por longitud para evitar conflictos de substring.
    """
    lower = name.lower()
    for cls in sorted(CLASSES, key=len, reverse=True):
        if cls.lower() in lower:
            return cls
    return None


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    """Recorre el directorio origen y agrupa rutas por clase detectada."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for p in source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}:
            cls = detect_class(p.name)
            if cls:
                grouped[cls].append(p)
    return grouped


def split_and_copy(source_dir: Path, target_dir: Path, seed: int, mode: str) -> None:
    """Crea carpetas train/val/test y copia/mueve archivos por clase."""
    rng = random.Random(seed)
    grouped = collect_images(source_dir)

    for split, _ in SPLITS:
        for cls in CLASSES:
            (target_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        files = grouped.get(cls, [])
        if not files:
            continue
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, items in split_files.items():
            for src in items:
                dst = target_dir / split / cls / src.name
                if mode == "move":
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(src, dst)
        print(f"{cls}: train={len(split_files['train'])}, val={len(split_files['val'])}, test={len(split_files['test'])}")


def main() -> None:
    """Punto de entrada CLI para particionar el dataset."""
    parser = argparse.ArgumentParser(description="Particion estratificada 70/15/15 por clase.")
    parser.add_argument("--source", type=Path, default=Path("images"))
    parser.add_argument("--target", type=Path, default=Path("dataset_701515"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"No existe: {args.source}")
    split_and_copy(args.source, args.target, args.seed, args.mode)


if __name__ == "__main__":
    main()
