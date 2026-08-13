import base64
import re
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.core.errors import InvalidImageError
from app.schemas.common import IssueCode

_DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.*)$", re.DOTALL)


def _check_size(raw: bytes) -> None:
    if len(raw) == 0:
        raise InvalidImageError(IssueCode.INVALID_IMAGE, "La imagen enviada está vacía.")
    max_bytes = int(settings.face_max_image_mb * 1024 * 1024)
    if len(raw) > max_bytes:
        raise InvalidImageError(
            IssueCode.IMAGE_TOO_LARGE,
            f"La imagen supera el límite de {settings.face_max_image_mb}MB.",
        )


async def bytes_from_upload(image: UploadFile) -> bytes:
    raw = await image.read()
    _check_size(raw)
    return raw


def bytes_from_base64_or_data_url(value: str) -> bytes:
    if not value or not value.strip():
        raise InvalidImageError(IssueCode.INVALID_IMAGE, "El campo 'image' está vacío.")
    match = _DATA_URL_RE.match(value.strip())
    b64_payload = match.group(1) if match else value.strip()
    try:
        raw = base64.b64decode(b64_payload, validate=True)
    except Exception as exc:
        raise InvalidImageError(IssueCode.INVALID_IMAGE, "El campo 'image' no es base64 válido.") from exc
    _check_size(raw)
    return raw


def decode_from_bytes(raw: bytes) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError(
            IssueCode.INVALID_IMAGE,
            "No se pudo decodificar la imagen. Verifique que el archivo no esté dañado.",
        )
    return img


def parse_optional_float(
    value,
    field_name: str,
    *,
    ge: Optional[float] = None,
    le: Optional[float] = None,
) -> Optional[float]:
    """Castea un campo de multipart/form-data (string) a float opcional, con validación de rango.

    Usado para campos que en JSON ya vienen tipados por pydantic (max_yaw, threshold, ...)
    pero que en multipart llegan siempre como string.
    """
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"El campo '{field_name}' debe ser numérico.")
    if ge is not None and parsed < ge:
        raise HTTPException(status_code=422, detail=f"El campo '{field_name}' debe ser >= {ge}.")
    if le is not None and parsed > le:
        raise HTTPException(status_code=422, detail=f"El campo '{field_name}' debe ser <= {le}.")
    return parsed
