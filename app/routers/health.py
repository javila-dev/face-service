from fastapi import APIRouter, Response, status

from app.config import settings
from app.services import model_registry

router = APIRouter(tags=["health"])


@router.get("/health", summary="Estado del servicio y del modelo cargado")
def health(response: Response):
    ready = model_registry.is_ready()
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ready else "not_ready",
        "model": settings.face_model_name,
        "version": settings.app_version,
        "backend": "insightface",
    }
