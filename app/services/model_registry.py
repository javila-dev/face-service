import logging
from typing import Optional

from insightface.app import FaceAnalysis

from app.config import settings
from app.core.errors import ModelNotReadyError

logger = logging.getLogger(__name__)

# El head de reconocimiento ArcFace usado por todos los paquetes buffalo_*
# de InsightFace produce embeddings de 512 dimensiones.
EMBEDDING_DIM = 512

_analyzer: Optional[FaceAnalysis] = None
_ready: bool = False


def load_model() -> None:
    """Carga el modelo una sola vez al arrancar el proceso (llamado desde el lifespan de FastAPI)."""
    global _analyzer, _ready
    try:
        analyzer = FaceAnalysis(
            name=settings.face_model_name,
            root=settings.face_models_root,
            providers=settings.onnx_providers,
        )
        ctx_id = 0 if settings.face_providers.lower() == "cuda" else -1
        analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
        _analyzer = analyzer
        _ready = True
        logger.info("Modelo '%s' cargado correctamente.", settings.face_model_name)
    except Exception:
        _analyzer = None
        _ready = False
        logger.exception("Error cargando el modelo '%s'.", settings.face_model_name)


def is_ready() -> bool:
    return _ready


def get_analyzer() -> FaceAnalysis:
    if not _ready or _analyzer is None:
        raise ModelNotReadyError()
    return _analyzer


def reset_for_testing(analyzer, ready: bool = True) -> None:
    """Hook exclusivo para tests: inyecta un analyzer falso sin cargar el modelo real."""
    global _analyzer, _ready
    _analyzer = analyzer
    _ready = ready
