import json
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, Request

from app.config import settings
from app.core.errors import EmbeddingError, InvalidImageError
from app.deps.auth import require_api_key
from app.schemas.common import IssueCode
from app.schemas.match import MatchImageInput, MatchResponse
from app.services import embedding, image_input, model_registry, quality

router = APIRouter(prefix="/v1", tags=["match"])


@router.post(
    "/match",
    response_model=MatchResponse,
    summary="Compara una foto en vivo contra un embedding de control (1:1)",
    description=(
        "Acepta multipart/form-data (campos 'image', 'embedding' como JSON-string, "
        "'threshold', 'max_yaw'/'max_pitch'/'max_roll' opcionales) o JSON con "
        "{'image', 'embedding', 'threshold', 'max_yaw', 'max_pitch', 'max_roll'}. "
        "El servicio decide match_passed comparando la similitud coseno contra el umbral. "
        "Mandar cualquiera de los límites de pose hace que bad_pose bloquee ok para esa request."
    ),
)
async def match(request: Request, _api_key: str = Depends(require_api_key)) -> MatchResponse:
    raw, embedding_data, threshold, max_yaw, max_pitch, max_roll = await _extract_payload(request)

    control_emb = embedding.list_to_embedding(embedding_data)
    threshold_value = threshold if threshold is not None else settings.face_match_threshold

    img = image_input.decode_from_bytes(raw)
    analyzer = model_registry.get_analyzer()
    faces = analyzer.get(img)
    result = quality.analyze(img, faces, max_yaw=max_yaw, max_pitch=max_pitch, max_roll=max_roll)

    if not result.ok:
        return MatchResponse(ok=False, threshold=threshold_value, issues=result.issues)

    live_emb = embedding.extract_embedding(result.face)
    score = embedding.cosine_similarity(live_emb, control_emb)
    passed = embedding.decide_match(score, threshold_value)

    return MatchResponse(
        ok=True,
        match_score=score,
        match_passed=passed,
        threshold=threshold_value,
        det_score=result.det_score,
        face_ratio=result.face_ratio,
        issues=result.issues,
    )


async def _extract_payload(
    request: Request,
) -> Tuple[bytes, List[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            raise InvalidImageError(IssueCode.INVALID_IMAGE, "Falta el campo 'image' en el form-data.")
        raw = await image_input.bytes_from_upload(upload)

        embedding_raw = form.get("embedding")
        if not embedding_raw:
            raise EmbeddingError(IssueCode.EMBEDDING_DIMENSION_MISMATCH, "Falta el campo 'embedding'.")
        try:
            embedding_data = json.loads(embedding_raw)
        except Exception as exc:
            raise EmbeddingError(
                IssueCode.EMBEDDING_DIMENSION_MISMATCH, "El campo 'embedding' no es JSON válido."
            ) from exc

        threshold = image_input.parse_optional_float(form.get("threshold"), "threshold", ge=-1.0, le=1.0)
        max_yaw = image_input.parse_optional_float(form.get("max_yaw"), "max_yaw", ge=0, le=90)
        max_pitch = image_input.parse_optional_float(form.get("max_pitch"), "max_pitch", ge=0, le=90)
        max_roll = image_input.parse_optional_float(form.get("max_roll"), "max_roll", ge=0, le=90)
        return raw, embedding_data, threshold, max_yaw, max_pitch, max_roll

    try:
        body = await request.json()
    except Exception as exc:
        raise InvalidImageError(IssueCode.INVALID_IMAGE, "El body no es JSON válido.") from exc

    payload = MatchImageInput.model_validate(body)
    raw = image_input.bytes_from_base64_or_data_url(payload.image)
    return raw, payload.embedding, payload.threshold, payload.max_yaw, payload.max_pitch, payload.max_roll
