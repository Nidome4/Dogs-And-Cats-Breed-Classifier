# Guia de Exposicion del Proyecto

Este documento resume la arquitectura del repositorio y responde las preguntas clave que ya aparecieron durante el desarrollo.

## 1) Estructura del repo

Raiz del proyecto (`C:\Users\berku\OneDrive\Desktop\archive`):

- `split_dataset.py`
- `balance_classes.py`
- `train_models.py`
- `hyperparameter_tuning.py`
- `inference.py`
- `ensemble_inference.py`
- `app.py`
- `organize_dataset.py` (script inicial de organizacion)
- `README.md`
- `requirements.txt`
- `src/`
- `models/`
- `dataset_701515/` (dataset particionado)
- `artifacts/` (salidas de entrenamiento/inferencia)

Subcarpetas clave:

- `src/ml_utils.py`: utilidades comunes de datos, metricas y guardado.
- `models/resnet50_model.py`: definicion de modelo ResNet50.
- `models/efficientnet_b0_model.py`: definicion de modelo EfficientNetB0.
- `models/mobilenetv3_model.py`: definicion de modelo MobileNetV3.

## 2) Organizacion y funcion de cada archivo

### A. Preparacion de datos

- `split_dataset.py`
  - Hace particion estratificada por clase.
  - Split actual: **70/15/15** (`train/val/test`).
  - Organiza en carpetas por clase.

- `balance_classes.py`
  - Balancea clases por **data augmentation offline**.
  - Copia originales y genera imagenes nuevas para clases minoritarias hasta igualar el target.
  - Recomendacion metodologica: aplicar solo a `train`, no a `val/test`.

- `organize_dataset.py`
  - Script inicial que organizaba en entrenamiento/validacion/testeo.
  - Quedo como referencia historica del proyecto.

### B. Entrenamiento y evaluacion

- `src/ml_utils.py`
  - **Data loader** principal con `make_datasets(...)`.
  - Carga imagenes desde carpetas con `tf.keras.utils.image_dataset_from_directory`.
  - Define `BATCH_SIZE = 32` (entrenamiento por lotes/batch training).
  - Calcula `class_weight` para balancear entrenamiento.
  - Evalua con `macro_f1` y `classification_report`.
  - Guarda artefactos (`best_model.keras`, `metadata.json`).

- `train_models.py`
  - Entrena y compara minimo 3 modelos:
    - MobileNetV2
    - EfficientNetB0
    - ResNet50
  - Usa class weights + early stopping.
  - Selecciona mejor modelo por `macro_f1`.
  - Guarda benchmark y reportes por modelo.
  - Guarda modelos individuales para ensemble en `artifacts/ensemble_models/`.

- `hyperparameter_tuning.py`
  - Implementa **optimizacion formal de hiperparametros** con `KerasTuner` (Random Search).
  - Busca combinaciones de:
    - backbone
    - learning rate
    - dropout
  - Guarda mejor modelo tunado y resultados en `artifacts/tuned/`.

### C. Inferencia y despliegue

- `inference.py`
  - Carga `artifacts/best_model.keras` y `metadata.json`.
  - Predice top-k clases para una imagen desde terminal.

- `ensemble_inference.py`
  - Carga multiples modelos guardados (`artifacts/ensemble_models/*.keras`).
  - Combina predicciones por promedio de probabilidades.
  - Entrega top-k del ensamble.

- `app.py`
  - API Flask de inferencia.
  - Endpoints:
    - `GET /health`
    - `POST /predict` (archivo en campo `file`).

### D. Documentacion y dependencias

- `README.md`
  - Guia operativa del pipeline y comandos.
- `requirements.txt`
  - Dependencias Python (TensorFlow, Flask, scikit-learn, keras-tuner, etc.).

## 3) Preguntas clave para la exposicion (respuestas directas)

### 3.1 Particion de datos

- Se usa **70/15/15** en `split_dataset.py`.
- Estructura final:
  - `train/<clase>/...`
  - `val/<clase>/...`
  - `test/<clase>/...`

### 3.2 Balanceo de clases

- Se hace con `balance_classes.py`.
- Si una clase tiene menos imagenes, se generan muestras nuevas por augmentation hasta igualar el numero objetivo.

### 3.3 Data loader y data generator

- **Data loader**: `make_datasets(...)` en `src/ml_utils.py`.
- **Data generator**:
  - Online: pipeline `tf.data` durante entrenamiento.
  - Offline: `balance_classes.py` generando archivos augmentados.

### 3.4 Batch training

- Si, hay entrenamiento por lotes.
- Batch actual: `BATCH_SIZE = 32` en `src/ml_utils.py`.

### 3.5 Metricas usadas

- Durante entrenamiento/evaluacion:
  - `accuracy`
  - `top3_acc`
  - `precision`
  - `recall`
- Evaluacion adicional:
  - `macro_f1`
  - `classification_report` por clase

### 3.6 Donde se ejecuta el modelo

- Entrenamiento: `train_models.py` (y tuning en `hyperparameter_tuning.py`).
- Inferencia local: `inference.py`.
- Inferencia por API: `app.py`.
- Ensamble: `ensemble_inference.py`.

### 3.7 Hiperparametros: si/no y donde

- Si, implementado en `hyperparameter_tuning.py` con KerasTuner.
- Salidas en:
  - `artifacts/tuner/`
  - `artifacts/tuned/`

### 3.8 Ensemble: si/no y donde

- Si, implementado.
- Modelos base guardados por `train_models.py`.
- Prediccion ensemble en `ensemble_inference.py`.

### 3.9 Se usa CSV para organizar imagenes?

- No.
- La organizacion principal es por carpetas (`train/val/test` por clase).
- Si se necesitara, se puede agregar un `labels.csv` como extension.

### 3.10 Serializacion y carga

- Serializacion:
  - `artifacts/best_model.keras`
  - `artifacts/metadata.json`
- Carga:
  - `inference.py` y `app.py` usan esos archivos para prediccion.

## 4) Flujo de ejecucion recomendado (demo)

1. Split 70/15/15:

```bash
python split_dataset.py --source images --target dataset_701515 --mode copy
```

2. Balancear train:

```bash
python balance_classes.py --input-dir dataset_701515/train --output-dir dataset_701515_balanced/train
```

3. Entrenar modelos base:

```bash
python train_models.py --data-dir dataset_701515 --out-dir artifacts --epochs 5
```

4. (Opcional) Tuning:

```bash
python hyperparameter_tuning.py --data-dir dataset_701515 --out-dir artifacts --max-trials 8 --epochs 5
```

5. Inferencia simple:

```bash
python inference.py --image ruta/a/imagen.jpg --artifacts artifacts --top-k 3
```

6. Inferencia con ensemble:

```bash
python ensemble_inference.py --image ruta/a/imagen.jpg --artifacts artifacts --top-k 3
```

7. API Flask:

```bash
python app.py
```

## 5) Mensaje final para defensa

El proyecto cumple un pipeline completo de clasificacion de imagenes: preparacion de datos, balanceo, entrenamiento comparativo con multiples arquitecturas, metricas robustas, optimizacion de hiperparametros, ensemble, serializacion y despliegue por API.
