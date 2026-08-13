from dataclasses import dataclass, field
from typing import Any, List, Optional

import cv2
import numpy as np

from app.config import settings
from app.schemas.common import Issue, IssueCode, make_issue

# Códigos que efectivamente bloquean ok=true. bad_pose queda fuera a propósito:
# no todos los paquetes de InsightFace exponen pose de forma confiable, así que
# se reporta como señal adicional pero nunca tumba la respuesta por sí sola.
_BLOCKING_CODES = {
    IssueCode.NO_FACE,
    IssueCode.MULTIPLE_FACES,
    IssueCode.LOW_CONFIDENCE,
    IssueCode.FACE_TOO_SMALL,
    IssueCode.BLURRY,
    IssueCode.TOO_DARK,
    IssueCode.TOO_BRIGHT,
    IssueCode.LOW_RESOLUTION,
    IssueCode.INVALID_IMAGE,
}


@dataclass
class QualityResult:
    ok: bool
    face: Optional[Any]
    det_score: float
    face_ratio: float
    laplacian_var: float
    brightness: float
    issues: List[Issue] = field(default_factory=list)


def _empty_result(issues: List[Issue]) -> QualityResult:
    return QualityResult(False, None, 0.0, 0.0, 0.0, 0.0, issues)


def analyze(img: np.ndarray, faces: list) -> QualityResult:
    h, w = img.shape[:2]

    if min(h, w) < settings.face_min_resolution:
        return _empty_result([make_issue(
            IssueCode.LOW_RESOLUTION,
            f"La imagen es demasiado pequeña ({w}x{h}px). Se requiere mínimo "
            f"{settings.face_min_resolution}x{settings.face_min_resolution}px.",
        )])

    if len(faces) == 0:
        return _empty_result([make_issue(IssueCode.NO_FACE)])

    if len(faces) > 1:
        return _empty_result([make_issue(IssueCode.MULTIPLE_FACES)])

    face = faces[0]
    x1, y1, x2, y2 = [int(c) for c in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    det_score = float(face.det_score)
    face_ratio = ((x2 - x1) * (y2 - y1)) / (w * h) * 100

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return _empty_result([make_issue(IssueCode.INVALID_IMAGE, "No se pudo recortar el rostro detectado.")])

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    issues: List[Issue] = []
    if det_score < settings.face_min_det_score:
        issues.append(make_issue(IssueCode.LOW_CONFIDENCE))
    if face_ratio < settings.face_min_face_ratio:
        issues.append(make_issue(IssueCode.FACE_TOO_SMALL))
    if laplacian_var < settings.face_min_laplacian_var:
        issues.append(make_issue(IssueCode.BLURRY))
    if brightness < settings.face_min_brightness:
        issues.append(make_issue(IssueCode.TOO_DARK))
    elif brightness > settings.face_max_brightness:
        issues.append(make_issue(IssueCode.TOO_BRIGHT))

    pose = getattr(face, "pose", None)
    if pose is not None:
        try:
            yaw = float(pose[1])
        except (TypeError, IndexError, ValueError):
            yaw = None
        if yaw is not None and abs(yaw) > 35:
            issues.append(make_issue(IssueCode.BAD_POSE))

    ok = not any(issue.code in _BLOCKING_CODES for issue in issues)
    return QualityResult(
        ok=ok,
        face=face if ok else None,
        det_score=det_score,
        face_ratio=face_ratio,
        laplacian_var=laplacian_var,
        brightness=brightness,
        issues=issues,
    )
