from __future__ import annotations
"""Entrena y compara 3 modelos de clasificacion; guarda el mejor por macro-F1.

Modelos evaluados:
- MobileNetV2
- EfficientNetB0
- ResNet50

Flujo para novatos:
1) Carga data (`train/val/test`) con el data loader.
2) Calcula class weights para balancear entrenamiento.
3) Entrena 3 modelos distintos.
4) Evalua en test con varias metricas.
5) Guarda el mejor modelo para inferencia.
6) Guarda cada modelo individual para poder hacer ensemble.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.ml_utils import (
    class_weights_from_ds,
    evaluate_macro_f1,
    make_datasets,
    make_model_factory,
    save_artifacts,
)

MODEL_NAMES = ["MobileNetV2", "EfficientNetB0", "ResNet50"]


def main() -> None:
    """Ejecuta entrenamiento, evaluacion y serializacion de artefactos."""
    parser = argparse.ArgumentParser(description="Entrena 3 modelos y guarda el mejor.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset_701515"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # Data loader principal del proyecto.
    train_ds, val_ds, test_ds, class_names = make_datasets(args.data_dir)
    # Balanceo por pesos de clase (no modifica imagenes; modifica la funcion de costo).
    cw = class_weights_from_ds(train_ds, len(class_names))

    results = []
    best = {"name": None, "f1": -1.0, "model": None}
    ensemble_dir = args.out_dir / "ensemble_models"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    for name in MODEL_NAMES:
        print(f"\n=== Entrenando {name} ===")
        model = make_model_factory(name)(len(class_names))
        # Early stopping para evitar sobreajuste y reducir tiempo de entrenamiento.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ]
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=cw,
            callbacks=callbacks,
            verbose=1,
        )
        eval_dict = model.evaluate(test_ds, return_dict=True, verbose=0)
        # Macro-F1 y reporte por clase para analizar rendimiento fino.
        macro_f1, report = evaluate_macro_f1(model, test_ds)

        row = {
            "model": name,
            "test_loss": float(eval_dict["loss"]),
            "test_accuracy": float(eval_dict.get("accuracy", 0.0)),
            "test_top3_acc": float(eval_dict.get("top3_acc", 0.0)),
            "macro_f1": macro_f1,
            "epochs_ran": len(hist.history.get("loss", [])),
        }
        results.append(row)

        (args.out_dir / f"classification_report_{name}.txt").parent.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"classification_report_{name}.txt").write_text(report, encoding="utf-8")

        # Guardar cada modelo entrenado (para ensemble por promedio de probabilidades).
        model.save(ensemble_dir / f"{name}.keras")

        if macro_f1 > best["f1"]:
            best = {"name": name, "f1": macro_f1, "model": model}

    save_artifacts(best["model"], class_names, args.out_dir, best["name"])
    (args.out_dir / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    ensemble_meta = {
        "model_files": [f"{n}.keras" for n in MODEL_NAMES],
        "class_names": class_names,
        "strategy": "average_probabilities",
    }
    (args.out_dir / "ensemble_metadata.json").write_text(json.dumps(ensemble_meta, indent=2), encoding="utf-8")

    print("\n=== Resumen ===")
    for r in results:
        print(r)
    print(f"\nModelo ganador: {best['name']} | macro_f1={best['f1']:.4f}")
    print(f"Modelos para ensemble guardados en: {ensemble_dir}")


if __name__ == "__main__":
    main()
