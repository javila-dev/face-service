from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import Issue


class MatchImageInput(BaseModel):
    image: str = Field(..., description="Imagen en base64 crudo o data-URL (data:image/jpeg;base64,...).")
    embedding: List[float] = Field(..., description="Embedding de control (512 floats) generado por /v1/enroll.")
    threshold: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Umbral opcional de esta request. Si se omite se usa FACE_MATCH_THRESHOLD.",
    )
    max_yaw: Optional[float] = Field(
        None, ge=0, le=90, description="Límite de yaw (giro) en grados para esta request. Mandarlo hace que bad_pose bloquee ok."
    )
    max_pitch: Optional[float] = Field(
        None, ge=0, le=90, description="Límite de pitch (inclinación vertical) en grados para esta request."
    )
    max_roll: Optional[float] = Field(
        None, ge=0, le=90, description="Límite de roll (inclinación lateral) en grados para esta request."
    )


class MatchResponse(BaseModel):
    ok: bool
    match_score: Optional[float] = Field(None, description="Similitud coseno en [-1, 1].")
    match_passed: Optional[bool] = None
    threshold: Optional[float] = None
    det_score: Optional[float] = None
    face_ratio: Optional[float] = None
    issues: List[Issue] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ok": True,
                    "match_score": 0.71,
                    "match_passed": True,
                    "threshold": 0.40,
                    "det_score": 0.95,
                    "face_ratio": 21.3,
                    "issues": [],
                },
                {
                    "ok": False,
                    "match_score": None,
                    "match_passed": None,
                    "threshold": 0.40,
                    "det_score": None,
                    "face_ratio": None,
                    "issues": [{"code": "no_face", "message": "No se detectó ningún rostro en la imagen."}],
                },
            ]
        }
    }
