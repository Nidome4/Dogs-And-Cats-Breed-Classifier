from __future__ import annotations
"""Optimizacion de hiperparametros con KerasTuner (Random Search).

Busca la mejor combinacion de:
- learning_rate
- dropout
- backbone (MobileNetV2 / EfficientNetB0 / ResNet50)

Seleccion por objetivo: `val_sparse_categorical_accuracy`.
Luego evalua en test y guarda el modelo tunado.
"""

import argparse
import json
from pathlib import Path

import keras_tuner as kt
import tensorflow as tf

from src.ml_utils import class_weights_from_ds, evaluate_macro_f1, make_datasets


class ImageHyperModel(kt.HyperModel):
    """Define el espacio de hiperparametros para clasificacion de imagenes."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        backbone_name = hp.Choice("backbone", ["MobileNetV2", "EfficientNetB0", "ResNet50"])
        dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
        lr = hp.Choice("learning_rate", [1e-4, 3e-4, 1e-3])

        if backbone_name == "MobileNetV2":
            base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        elif backbone_name == "EfficientNetB0":
            base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        else:
            base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

        base.trainable = False

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Hiperparametros con KerasTuner RandomSearch.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset_701515"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_names = make_datasets(args.data_dir)
    class_weights = class_weights_from_ds(train_ds, len(class_names))

    tuner = kt.RandomSearch(
        hypermodel=ImageHyperModel(num_classes=len(class_names)),
        objective="val_accuracy",
        max_trials=args.max_trials,
        overwrite=True,
        directory=str(args.out_dir / "tuner"),
        project_name="image_classifier",
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    eval_dict = best_model.evaluate(test_ds, return_dict=True, verbose=0)
    macro_f1, report = evaluate_macro_f1(best_model, test_ds)

    tuned_dir = args.out_dir / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)
    best_model.save(tuned_dir / "best_tuned_model.keras")

    result = {
        "best_hyperparameters": {
            "backbone": best_hp.get("backbone"),
            "dropout": best_hp.get("dropout"),
            "learning_rate": best_hp.get("learning_rate"),
        },
        "test_loss": float(eval_dict["loss"]),
        "test_accuracy": float(eval_dict.get("accuracy", 0.0)),
        "macro_f1": float(macro_f1),
    }

    (tuned_dir / "tuning_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (tuned_dir / "classification_report_tuned.txt").write_text(report, encoding="utf-8")

    print("\nMejores hiperparametros:")
    print(json.dumps(result["best_hyperparameters"], indent=2))
    print(f"\nTest accuracy: {result['test_accuracy']:.4f}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
