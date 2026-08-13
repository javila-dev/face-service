from enum import Enum
from typing import Optional

from pydantic import BaseModel


class IssueCode(str, Enum):
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    LOW_CONFIDENCE = "low_confidence"
    FACE_TOO_SMALL = "face_too_small"
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_RESOLUTION = "low_resolution"
    BAD_POSE = "bad_pose"
    INVALID_IMAGE = "invalid_image"
    IMAGE_TOO_LARGE = "image_too_large"
    UNSUPPORTED_FORMAT = "unsupported_format"
    EMBEDDING_DIMENSION_MISMATCH = "embedding_dimension_mismatch"


class Issue(BaseModel):
    code: IssueCode
    message: str


ISSUE_MESSAGES: dict[IssueCode, str] = {
    IssueCode.NO_FACE: "No se detectó ningún rostro en la imagen.",
    IssueCode.MULTIPLE_FACES: "Se detectó más de un rostro en la imagen.",
    IssueCode.LOW_CONFIDENCE: "La confianza de detección del rostro es demasiado baja.",
    IssueCode.FACE_TOO_SMALL: "El rostro ocupa muy poco espacio en la imagen.",
    IssueCode.BLURRY: "La imagen está borrosa.",
    IssueCode.TOO_DARK: "La imagen está demasiado oscura.",
    IssueCode.TOO_BRIGHT: "La imagen está demasiado brillante.",
    IssueCode.LOW_RESOLUTION: "La resolución de la imagen es demasiado baja.",
    IssueCode.BAD_POSE: "El ángulo del rostro no es adecuado.",
    IssueCode.INVALID_IMAGE: "No se pudo leer la imagen enviada.",
    IssueCode.IMAGE_TOO_LARGE: "La imagen supera el tamaño máximo permitido.",
    IssueCode.UNSUPPORTED_FORMAT: "El formato de imagen no es compatible.",
    IssueCode.EMBEDDING_DIMENSION_MISMATCH: "El embedding recibido no tiene la dimensión esperada.",
}


def make_issue(code: IssueCode, message: Optional[str] = None) -> Issue:
    return Issue(code=code, message=message or ISSUE_MESSAGES[code])
