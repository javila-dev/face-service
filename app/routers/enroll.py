from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from app.core.errors import InvalidImageError
from app.deps.auth import require_api_key
from app.schemas.common import IssueCode
from app.schemas.enroll import EnrollImageInput, EnrollResponse
from app.services import embedding, image_input, model_registry, quality

router = APIRouter(prefix="/v1", tags=["enroll"])


@dataclass
class EnrollParams:
    raw: bytes
    max_yaw: Optional[float] = None
    max_pitch: Optional[float] = None
    max_roll: Optional[float] = None
    min_det_score: Optional[float] = None
    min_face_ratio: Optional[float] = None
    min_laplacian_var: Optional[float] = None
    min_brightness: Optional[float] = None
    max_brightness: Optional[float] = None
    min_resolution: Optional[int] = None


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    summary="Genera el embedding de enrolamiento a partir de una foto",
    description=(
        "Acepta multipart/form-data o JSON con el campo 'image' y, opcionalmente, "
        "'max_yaw'/'max_pitch'/'max_roll' y los umbrales de calidad "
        "'min_det_score'/'min_face_ratio'/'min_laplacian_var'/'min_brightness'/"
        "'max_brightness'/'min_resolution'. Cualquier umbral que no se mande toma "
        "el valor configurado por variable de entorno. Mandar cualquiera de los "
        "límites de pose hace que bad_pose bloquee ok para esa request."
    ),
)
async def enroll(request: Request, _api_key: str = Depends(require_api_key)) -> EnrollResponse:
    params = await _extract_payload(request)
    img = image_input.decode_from_bytes(params.raw)

    analyzer = model_registry.get_analyzer()
    faces = await run_in_threadpool(analyzer.get, img)
    result = quality.analyze(
        img,
        faces,
        max_yaw=params.max_yaw,
        max_pitch=params.max_pitch,
        max_roll=params.max_roll,
        min_det_score=params.min_det_score,
        min_face_ratio=params.min_face_ratio,
        min_laplacian_var=params.min_laplacian_var,
        min_brightness=params.min_brightness,
        max_brightness=params.max_brightness,
        min_resolution=params.min_resolution,
    )

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


async def _extract_payload(request: Request) -> EnrollParams:
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            raise InvalidImageError(IssueCode.INVALID_IMAGE, "Falta el campo 'image' en el form-data.")
        raw = await image_input.bytes_from_upload(upload)

        return EnrollParams(
            raw=raw,
            max_yaw=image_input.parse_optional_float(form.get("max_yaw"), "max_yaw", ge=0, le=90),
            max_pitch=image_input.parse_optional_float(form.get("max_pitch"), "max_pitch", ge=0, le=90),
            max_roll=image_input.parse_optional_float(form.get("max_roll"), "max_roll", ge=0, le=90),
            min_det_score=image_input.parse_optional_float(
                form.get("min_det_score"), "min_det_score", ge=0, le=1
            ),
            min_face_ratio=image_input.parse_optional_float(
                form.get("min_face_ratio"), "min_face_ratio", ge=0, le=100
            ),
            min_laplacian_var=image_input.parse_optional_float(
                form.get("min_laplacian_var"), "min_laplacian_var", ge=0
            ),
            min_brightness=image_input.parse_optional_float(
                form.get("min_brightness"), "min_brightness", ge=0, le=255
            ),
            max_brightness=image_input.parse_optional_float(
                form.get("max_brightness"), "max_brightness", ge=0, le=255
            ),
            min_resolution=image_input.parse_optional_int(
                form.get("min_resolution"), "min_resolution", ge=1
            ),
        )

    try:
        body = await request.json()
    except Exception as exc:
        raise InvalidImageError(IssueCode.INVALID_IMAGE, "El body no es JSON válido.") from exc

    payload = EnrollImageInput.model_validate(body)
    raw = image_input.bytes_from_base64_or_data_url(payload.image)
    return EnrollParams(
        raw=raw,
        max_yaw=payload.max_yaw,
        max_pitch=payload.max_pitch,
        max_roll=payload.max_roll,
        min_det_score=payload.min_det_score,
        min_face_ratio=payload.min_face_ratio,
        min_laplacian_var=payload.min_laplacian_var,
        min_brightness=payload.min_brightness,
        max_brightness=payload.max_brightness,
        min_resolution=payload.min_resolution,
    )
