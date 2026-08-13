from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.config import settings
from app.core.errors import EmbeddingError, InvalidImageError, ModelNotReadyError
from app.routers import enroll, health, match
from app.services import model_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_registry.load_model()
    yield


app = FastAPI(
    title="Face Recognition Service",
    description=(
        "Microservicio de reconocimiento facial 1:1, reutilizable por varias apps. "
        "Stateless: no guarda personas, solo genera y compara embeddings. "
        "No conoce dominio de negocio."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(enroll.router)
app.include_router(match.router)


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request: Request, exc: InvalidImageError):
    return JSONResponse(status_code=400, content={"detail": exc.message, "code": exc.code})


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: EmbeddingError):
    return JSONResponse(status_code=422, content={"detail": exc.message, "code": exc.code})


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    return JSONResponse(status_code=503, content={"detail": exc.message})


@app.exception_handler(PydanticValidationError)
async def pydantic_validation_handler(request: Request, exc: PydanticValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Datos de entrada inválidos.", "errors": exc.errors()},
    )
