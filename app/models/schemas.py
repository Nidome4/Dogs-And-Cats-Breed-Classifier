from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    label: str = Field(..., description="Breed label")
    confidence: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    breed: str
    probability: float = Field(..., ge=0, le=1)
    top5: list[PredictionItem]
    inference_time_ms: int = Field(..., ge=0)
    model_used: str


class URLRequest(BaseModel):
    url: str


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    available_models: list[str]
