# Proyecto Clasificacion de Razas (TensorFlow + Flask)

## Objetivo del proyecto
Entrenar un clasificador multiclase de razas (gatos/perros), comparando varios modelos, guardando el mejor, y exponer inferencia por script y por API Flask.

## Conceptos clave (version novatos)
- Data loader: componente que lee datos desde disco en batches.
  En este proyecto: `make_datasets(...)` en `src/ml_utils.py`.
- Data generator: componente que crea/transforma datos.
  Aqui tienes 2 formas:
  1) Online en entrenamiento con `tf.data`.
  2) Offline con `balance_classes.py` (genera nuevas imagenes augmentadas y las guarda).
- Class imbalance: cuando unas clases tienen muchas mas imagenes que otras.
- Class weights: pesos por clase para que el entrenamiento no ignore clases pequenas.
- Macro-F1: metrica que da el mismo peso a todas las clases (util con desbalance).
- Hyperparameter tuning: busqueda automatica de mejor configuracion de entrenamiento.
- Ensemble: combinar varios modelos para una prediccion mas robusta.

## Flujo recomendado
1. Partir dataset en `train/val/test` con 70/15/15.
2. Balancear SOLO `train` con data augmentation.
3. Entrenar 3 modelos y seleccionar el mejor por macro-F1.
4. (Opcional) Hacer tuning de hiperparametros.
5. Usar inferencia simple (mejor modelo) o ensemble.

## 1) Crear split 70/15/15
```bash
python split_dataset.py --source images --target dataset_701515 --mode copy
```

## 2) Balancear clases (solo train)
```bash
python balance_classes.py --input-dir dataset_701515/train --output-dir dataset_701515_balanced/train
```

## 3) Entrenar y comparar 3 modelos
```bash
python train_models.py --data-dir dataset_701515 --out-dir artifacts --epochs 5
```

Salidas nuevas importantes:
- `artifacts/ensemble_models/*.keras`
- `artifacts/ensemble_metadata.json`

## 4) Optimizacion de hiperparametros (KerasTuner)
```bash
python hyperparameter_tuning.py --data-dir dataset_701515 --out-dir artifacts --max-trials 8 --epochs 5
```

Salidas:
- `artifacts/tuned/best_tuned_model.keras`
- `artifacts/tuned/tuning_result.json`
- `artifacts/tuned/classification_report_tuned.txt`

## 5) Inferencia por modelo ganador
```bash
python inference.py --image ruta/a/imagen.jpg --artifacts artifacts --top-k 3
```

## 6) Inferencia por ensemble
```bash
python ensemble_inference.py --image ruta/a/imagen.jpg --artifacts artifacts --top-k 3
```

## 7) API Flask
```bash
python app.py
```

### Probar API
```bash
curl -X POST -F "file=@ruta/a/imagen.jpg" http://localhost:5000/predict
```
