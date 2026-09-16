from dataclasses import dataclass, field
from typing import Any, List, Optional

import cv2
import numpy as np

from app.config import settings
from app.schemas.common import Issue, IssueCode, make_issue

# Códigos que siempre bloquean ok=true. bad_pose se agrega dinámicamente en
# analyze() solo cuando el chequeo de pose está en modo estricto (ver
# strict_pose más abajo) — por default queda fuera porque no todos los
# paquetes de InsightFace exponen pose de forma confiable.
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


def analyze(
    img: np.ndarray,
    faces: list,
    *,
    max_yaw: Optional[float] = None,
    max_pitch: Optional[float] = None,
    max_roll: Optional[float] = None,
    min_det_score: Optional[float] = None,
    min_face_ratio: Optional[float] = None,
    min_laplacian_var: Optional[float] = None,
    min_brightness: Optional[float] = None,
    max_brightness: Optional[float] = None,
    min_resolution: Optional[int] = None,
) -> QualityResult:
    eff_min_det_score = min_det_score if min_det_score is not None else settings.face_min_det_score
    eff_min_face_ratio = min_face_ratio if min_face_ratio is not None else settings.face_min_face_ratio
    eff_min_laplacian_var = min_laplacian_var if min_laplacian_var is not None else settings.face_min_laplacian_var
    eff_min_brightness = min_brightness if min_brightness is not None else settings.face_min_brightness
    eff_max_brightness = max_brightness if max_brightness is not None else settings.face_max_brightness
    eff_min_resolution = min_resolution if min_resolution is not None else settings.face_min_resolution

    h, w = img.shape[:2]

    if min(h, w) < eff_min_resolution:
        return _empty_result([make_issue(
            IssueCode.LOW_RESOLUTION,
            f"La imagen es demasiado pequeña ({w}x{h}px). Se requiere mínimo "
            f"{eff_min_resolution}x{eff_min_resolution}px.",
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
    if det_score < eff_min_det_score:
        issues.append(make_issue(IssueCode.LOW_CONFIDENCE))
    if face_ratio < eff_min_face_ratio:
        issues.append(make_issue(IssueCode.FACE_TOO_SMALL))
    if laplacian_var < eff_min_laplacian_var:
        issues.append(make_issue(IssueCode.BLURRY))
    if brightness < eff_min_brightness:
        issues.append(make_issue(IssueCode.TOO_DARK))
    elif brightness > eff_max_brightness:
        issues.append(make_issue(IssueCode.TOO_BRIGHT))

    eff_max_yaw = max_yaw if max_yaw is not None else settings.face_max_yaw
    eff_max_pitch = max_pitch if max_pitch is not None else settings.face_max_pitch
    eff_max_roll = max_roll if max_roll is not None else settings.face_max_roll

    pose = getattr(face, "pose", None)
    if pose is not None:
        try:
            p_pitch, p_yaw, p_roll = float(pose[0]), float(pose[1]), float(pose[2])
        except (TypeError, IndexError, ValueError):
            p_pitch = p_yaw = p_roll = None

        if p_yaw is not None:
            exceeded = []
            if eff_max_yaw is not None and abs(p_yaw) > eff_max_yaw:
                exceeded.append(f"yaw={p_yaw:.1f}°>{eff_max_yaw:.1f}°")
            if eff_max_pitch is not None and abs(p_pitch) > eff_max_pitch:
                exceeded.append(f"pitch={p_pitch:.1f}°>{eff_max_pitch:.1f}°")
            if eff_max_roll is not None and abs(p_roll) > eff_max_roll:
                exceeded.append(f"roll={p_roll:.1f}°>{eff_max_roll:.1f}°")
            if exceeded:
                issues.append(make_issue(
                    IssueCode.BAD_POSE,
                    "El ángulo del rostro no es adecuado (" + ", ".join(exceeded) + ").",
                ))

    strict_pose = settings.face_pose_blocking or any(
        v is not None for v in (max_yaw, max_pitch, max_roll)
    )
    blocking_codes = _BLOCKING_CODES | ({IssueCode.BAD_POSE} if strict_pose else set())
    ok = not any(issue.code in blocking_codes for issue in issues)
    return QualityResult(
        ok=ok,
        face=face if ok else None,
        det_score=det_score,
        face_ratio=face_ratio,
        laplacian_var=laplacian_var,
        brightness=brightness,
        issues=issues,
    )
