from __future__ import annotations
"""Modelo base MobileNetV3 (Small/Large) para clasificacion multiclase."""

import tensorflow as tf


def build_mobilenetv3(
    num_classes: int,
    variant: str = "large",
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout: float = 0.2,
) -> tf.keras.Model:
    """Construye un clasificador con MobileNetV3Large o MobileNetV3Small.

    Args:
        num_classes: numero de clases de salida.
        variant: "large" o "small".
    """
    v = variant.lower().strip()
    if v == "small":
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        model_name = "mobilenetv3_small_classifier"
    else:
        base = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        model_name = "mobilenetv3_large_classifier"

    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name=model_name)


def compile_mobilenetv3(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compila el modelo para clasificacion multiclase."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model
