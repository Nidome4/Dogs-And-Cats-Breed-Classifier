# Oxford-IIIT Pet Classifier

Aplicacion web + API para clasificacion de mascotas por raza.

## Modelos implementados (solo inferencia)
- EfficientNet-B0
- ResNet50
- MobileNetV3

## Caracteristicas
- Subida de imagenes
- Clasificacion por URL
- Selector de modelo
- Respuesta con:
  - Raza predicha
  - Probabilidad
  - Top 5
  - Tiempo de inferencia
  - Modelo usado

## Arquitectura

app/
- api/
- services/
- core/
- models/
- templates/
- static/
- main.py

## Pesos
- Los pesos se almacenan en `trained_models/`.
- En primera carga de un modelo, si no existe localmente, se descarga automaticamente.
- Puedes configurar la fuente de cada modelo con:
  - `PET_MODEL_SOURCE_EFFICIENTNET_B0`
  - `PET_MODEL_SOURCE_RESNET50`
  - `PET_MODEL_SOURCE_MOBILENETV3`

## Ejecutar

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abrir: http://localhost:8000

## Nota
No se implementa entrenamiento, fine-tuning ni pipelines de entrenamiento.
