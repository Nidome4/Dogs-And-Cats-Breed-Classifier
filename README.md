# Oxford-IIIT Pet Classifier

Aplicacion web + API para clasificacion de razas de perros y gatos. El proyecto incluye una app principal de inferencia y una fase de entrenamiento separada en `train_phase/`.

## App principal

La app principal se ejecuta con FastAPI desde `app/main.py` y expone una interfaz web para probar imagenes por archivo o URL.

### Modelos disponibles

Los modelos de la app principal se configuran en `app/core/config.py` y se corren desde el servicio `app/services/classifier_service.py`, usando `trained_models/` como directorio local de trabajo.

- EfficientNet-B0
- ResNet50
- MobileNetV3

### Caracteristicas

- Subida de imagenes
- Clasificacion por URL
- Selector de modelo
- Respuesta con raza predicha, probabilidad, top 5, tiempo de inferencia y modelo usado
- Endpoints para salud, listado de modelos y prediccion

### Ejecutar app principal

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abrir: http://localhost:8000

## Fase de entrenamiento

La carpeta `train_phase/` contiene el flujo para preparar datos, entrenar, comparar modelos y ejecutar inferencia con los artefactos generados por esa fase.

### Contenido principal

- `split_dataset.py`: separa el dataset en `train`, `val` y `test` con proporcion 70/15/15.
- `balance_classes.py`: balancea el split de entrenamiento con data augmentation.
- `data_generator.py`: utilidades para generar y revisar lotes de imagenes.
- `train_models.py`: entrena y compara MobileNetV2, EfficientNetB0 y ResNet50.
- `hyperparameter_tuning.py`: ejecuta busqueda de hiperparametros con KerasTuner.
- `inference.py`: corre inferencia desde `train_phase/artifacts/best_model.keras`.
- `ensemble_inference.py`: corre inferencia combinando modelos desde `train_phase/artifacts/ensemble_models/`.
- `app.py`: API Flask para inferencia desde los artefactos de `train_phase/artifacts/`.
- `src/ml_utils.py`: carga datasets, calcula class weights, evalua macro-F1 y guarda artefactos.
- `models/`: constructores de arquitecturas usadas durante entrenamiento.

### Flujo recomendado

Ejecutar los comandos desde `train_phase/`:

```bash
cd train_phase
python split_dataset.py --source images --target dataset_701515 --mode copy
python balance_classes.py --input-dir dataset_701515/train --output-dir dataset_701515_balanced/train
python train_models.py --data-dir dataset_701515 --out-dir artifacts --epochs 5
```

Salidas principales:

- `artifacts/best_model.keras`
- `artifacts/metadata.json`
- `artifacts/benchmark.json`
- `artifacts/classification_report_*.txt`
- `artifacts/ensemble_models/*.keras`
- `artifacts/ensemble_metadata.json`

### Inferencia desde la fase de entrenamiento

El mejor modelo generado por entrenamiento se corre desde `train_phase/artifacts/best_model.keras`:

```bash
cd train_phase
python inference.py --image ruta/a/imagen.jpg --artifacts artifacts --top-k 3
```

El ensemble se corre desde `train_phase/artifacts/ensemble_models/`:

```bash
cd train_phase
python ensemble_inference.py --image ruta/a/imagen.jpg --artifacts artifacts --top-k 3
```

La API Flask de esta fase tambien usa los artefactos ubicados en `train_phase/artifacts/`:

```bash
cd train_phase
python app.py
```

Probar API Flask:

```bash
curl -X POST -F "file=@ruta/a/imagen.jpg" http://localhost:5000/predict
```

## Estructura general

```text
app/
  api/
  core/
  models/
  services/
  static/
  templates/
  main.py
trained_models/
train_phase/
  artifacts/
  dataset_701515/
  models/
  src/
  app.py
  train_models.py
```
