from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.models.schemas import HealthResponse, PredictionItem, PredictionResponse, URLRequest
from app.services.classifier_service import get_classifier_service
from app.services.image_service import load_image_from_upload, load_image_from_url

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/health", response_model=HealthResponse)
async def health():
    service = get_classifier_service()
    return HealthResponse(status="healthy", available_models=service.available_models())


@router.get("/models")
async def list_models():
    service = get_classifier_service()
    return {"models": service.available_models()}


@router.post("/predict/file", response_model=PredictionResponse)
async def predict_file(
    file: UploadFile = File(...),
    model: str = Query("resnet50", description="efficientnet_b0 | resnet50 | mobilenetv3 | tmp | stage-2 | vit_b32"),
):
    image = await load_image_from_upload(file)
    service = get_classifier_service()
    try:
        result = service.predict(image, model)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PredictionResponse(
        breed=result["breed"],
        probability=result["probability"],
        top5=[PredictionItem(label=p["label"], confidence=p["confidence"]) for p in result["top5"]],
        inference_time_ms=result["inference_time_ms"],
        model_used=result["model_used"],
    )


@router.post("/predict/url", response_model=PredictionResponse)
async def predict_url(
    payload: URLRequest,
    model: str = Query("resnet50", description="efficientnet_b0 | resnet50 | mobilenetv3 | tmp | stage-2 | vit_b32"),
):
    image = load_image_from_url(payload.url)
    service = get_classifier_service()
    try:
        result = service.predict(image, model)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PredictionResponse(
        breed=result["breed"],
        probability=result["probability"],
        top5=[PredictionItem(label=p["label"], confidence=p["confidence"]) for p in result["top5"]],
        inference_time_ms=result["inference_time_ms"],
        model_used=result["model_used"],
    )


