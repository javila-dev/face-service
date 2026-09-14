from typing import Optional, Tuple

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from app.core.errors import InvalidImageError
from app.deps.auth import require_api_key
from app.schemas.common import IssueCode
from app.schemas.enroll import EnrollImageInput, EnrollResponse
from app.services import embedding, image_input, model_registry, quality

router = APIRouter(prefix="/v1", tags=["enroll"])


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    summary="Genera el embedding de enrolamiento a partir de una foto",
    description=(
        "Acepta multipart/form-data (campos 'image', 'max_yaw'/'max_pitch'/'max_roll' "
        "opcionales) o JSON con {'image', 'max_yaw', 'max_pitch', 'max_roll'}. Valida "
        "calidad de la foto y, si pasa, devuelve el embedding L2-normalizado. Mandar "
        "cualquiera de los límites de pose hace que bad_pose bloquee ok para esa request."
    ),
)
async def enroll(request: Request, _api_key: str = Depends(require_api_key)) -> EnrollResponse:
    raw, max_yaw, max_pitch, max_roll = await _extract_payload(request)
    img = image_input.decode_from_bytes(raw)

    analyzer = model_registry.get_analyzer()
    faces = await run_in_threadpool(analyzer.get, img)
    result = quality.analyze(img, faces, max_yaw=max_yaw, max_pitch=max_pitch, max_roll=max_roll)

    if not result.ok:
        return EnrollResponse(ok=False, issues=result.issues)

    emb = embedding.extract_embedding(result.face)
    return EnrollResponse(
        ok=True,
        embedding=embedding.embedding_to_list(emb),
        det_score=result.det_score,
        face_ratio=result.face_ratio,
        laplacian_var=result.laplacian_var,
        brightness=result.brightness,
        issues=result.issues,
    )


async def _extract_payload(request: Request) -> Tuple[bytes, Optional[float], Optional[float], Optional[float]]:
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            raise InvalidImageError(IssueCode.INVALID_IMAGE, "Falta el campo 'image' en el form-data.")
        raw = await image_input.bytes_from_upload(upload)

        max_yaw = image_input.parse_optional_float(form.get("max_yaw"), "max_yaw", ge=0, le=90)
        max_pitch = image_input.parse_optional_float(form.get("max_pitch"), "max_pitch", ge=0, le=90)
        max_roll = image_input.parse_optional_float(form.get("max_roll"), "max_roll", ge=0, le=90)
        return raw, max_yaw, max_pitch, max_roll

    try:
        body = await request.json()
    except Exception as exc:
        raise InvalidImageError(IssueCode.INVALID_IMAGE, "El body no es JSON válido.") from exc

    payload = EnrollImageInput.model_validate(body)
    raw = image_input.bytes_from_base64_or_data_url(payload.image)
    return raw, payload.max_yaw, payload.max_pitch, payload.max_roll
