from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import Issue


class EnrollImageInput(BaseModel):
    image: str = Field(..., description="Imagen en base64 crudo o data-URL (data:image/jpeg;base64,...).")
    max_yaw: Optional[float] = Field(
        None, ge=0, le=90, description="Límite de yaw (giro) en grados para esta request. Mandarlo hace que bad_pose bloquee ok."
    )
    max_pitch: Optional[float] = Field(
        None, ge=0, le=90, description="Límite de pitch (inclinación vertical) en grados para esta request."
    )
    max_roll: Optional[float] = Field(
        None, ge=0, le=90, description="Límite de roll (inclinación lateral) en grados para esta request."
    )
    min_det_score: Optional[float] = Field(
        None, ge=0, le=1, description="Confianza mínima de detección para esta request. Si se omite se usa FACE_MIN_DET_SCORE."
    )
    min_face_ratio: Optional[float] = Field(
        None, ge=0, le=100, description="Porcentaje mínimo del frame que debe ocupar el rostro. Si se omite se usa FACE_MIN_FACE_RATIO."
    )
    min_laplacian_var: Optional[float] = Field(
        None, ge=0, description="Varianza mínima del laplaciano (nitidez) para esta request. Si se omite se usa FACE_MIN_LAPLACIAN_VAR."
    )
    min_brightness: Optional[float] = Field(
        None, ge=0, le=255, description="Brillo mínimo aceptado para esta request. Si se omite se usa FACE_MIN_BRIGHTNESS."
    )
    max_brightness: Optional[float] = Field(
        None, ge=0, le=255, description="Brillo máximo aceptado para esta request. Si se omite se usa FACE_MAX_BRIGHTNESS."
    )
    min_resolution: Optional[int] = Field(
        None, ge=1, description="Resolución mínima (lado más chico, en px) para esta request. Si se omite se usa FACE_MIN_RESOLUTION."
    )


class EnrollResponse(BaseModel):
    ok: bool
    embedding: Optional[List[float]] = Field(
        None,
        description="Embedding L2-normalizado de 512 dimensiones. Solo presente si ok=true.",
    )
    det_score: Optional[float] = None
    face_ratio: Optional[float] = Field(None, description="Porcentaje del frame ocupado por el rostro.")
    laplacian_var: Optional[float] = Field(None, description="Varianza del laplaciano del rostro (medida de nitidez).")
    brightness: Optional[float] = None
    issues: List[Issue] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ok": True,
                    "embedding": [0.0123, -0.0456, 0.0789],
                    "det_score": 0.93,
                    "face_ratio": 18.4,
                    "laplacian_var": 342.1,
                    "brightness": 128.5,
                    "issues": [],
                },
                {
                    "ok": False,
                    "embedding": None,
                    "det_score": None,
                    "face_ratio": None,
                    "laplacian_var": None,
                    "brightness": None,
                    "issues": [{"code": "blurry", "message": "La imagen está borrosa."}],
                },
            ]
        }
    }
