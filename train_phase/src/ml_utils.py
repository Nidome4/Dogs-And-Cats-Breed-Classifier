from __future__ import annotations
"""Utilidades de ML para clasificacion multiclase de imagenes con TensorFlow.

Incluye:
- carga de datasets (train/val/test),
- construccion/compilacion de modelos,
- balanceo por class weights,
- evaluacion con macro-F1 y reporte de clasificacion,
- guardado de artefactos del mejor modelo.

Glosario rapido para principiantes:
- Data loader: bloque que lee imagenes desde disco y las entrega en lotes.
  Aqui es `make_datasets(...)`.
- Data generator: bloque que va produciendo datos durante el entrenamiento.
  En este proyecto, `image_dataset_from_directory` + `tf.data` cumplen ese papel.
- Class weights: pesos por clase para que el modelo no favorezca clases con
  mas ejemplos.
"""

import json
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def make_datasets(data_dir: Path, seed: int = 42):
    """Carga datasets desde carpetas `train`, `val` y `test`.

    Importante para novatos:
    - `train`: el modelo aprende aqui.
    - `val`: se usa para validar durante entrenamiento (no aprende de este split).
    - `test`: se usa al final para medir rendimiento real.
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, label_mode="int", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=seed
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, label_mode="int", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, label_mode="int", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    # `prefetch` mejora rendimiento: mientras GPU/CPU entrena un batch,
    # TensorFlow va preparando el siguiente.
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def build_head(base: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    """Agrega la cabeza de clasificacion sobre una base preentrenada."""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def compile_model(model: tf.keras.Model) -> None:
    """Compila el modelo con perdida multiclase y metricas clave."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),  # acierto exacto top-1
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),  # acierto dentro del top-3
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def make_model_factory(name: str) -> Callable[[int], tf.keras.Model]:
    """Devuelve una funcion constructora segun nombre del backbone."""
    if name == "MobileNetV2":
        def _f(nc: int):
            base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
            base.trainable = False
            model = build_head(base, nc)
            compile_model(model)
            return model
        return _f
    if name == "EfficientNetB0":
        def _f(nc: int):
            base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
            base.trainable = False
            model = build_head(base, nc)
            compile_model(model)
            return model
        return _f

    def _f(nc: int):
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        base.trainable = False
        model = build_head(base, nc)
        compile_model(model)
        return model
    return _f


def class_weights_from_ds(train_ds, num_classes: int):
    """Calcula class weights balanceados a partir del split de entrenamiento.

    Si una clase tiene pocas imagenes, su peso aumenta y el error en esa clase
    "cuesta mas" al optimizador.
    """
    labels = []
    for _, y in train_ds:
        labels.extend(y.numpy().tolist())
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=np.array(labels))
    return {i: float(w) for i, w in enumerate(weights)}


def evaluate_macro_f1(model: tf.keras.Model, test_ds) -> tuple[float, str]:
    """Evalua macro-F1 y genera `classification_report` en test.

    Macro-F1 promedia todas las clases por igual, por eso es util cuando hay
    desbalance.
    """
    y_true, y_pred = [], []
    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(y.numpy().tolist())
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=4)
    return float(f1), report


def save_artifacts(best_model, class_names, out_dir: Path, winner_name: str):
    """Guarda modelo ganador y metadata necesaria para inferencia."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_model.keras"
    best_model.save(model_path)
    meta = {
        "winner_model": winner_name,
        "class_names": class_names,
        "input_size": [224, 224],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

