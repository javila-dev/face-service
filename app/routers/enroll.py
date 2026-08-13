from fastapi import APIRouter, Depends, Request

from app.deps.auth import require_api_key
from app.schemas.enroll import EnrollResponse
from app.services import embedding, image_input, model_registry, quality

router = APIRouter(prefix="/v1", tags=["enroll"])


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    summary="Genera el embedding de enrolamiento a partir de una foto",
    description=(
        "Acepta multipart/form-data (campo 'image') o JSON con "
        "{'image': '<base64 o data-URL>'}. Valida calidad de la foto y, si pasa, "
        "devuelve el embedding L2-normalizado para que la app cliente lo guarde."
    ),
)
async def enroll(request: Request, _api_key: str = Depends(require_api_key)) -> EnrollResponse:
    raw = await image_input.extract_image_bytes(request)
    img = image_input.decode_from_bytes(raw)

    analyzer = model_registry.get_analyzer()
    faces = analyzer.get(img)
    result = quality.analyze(img, faces)

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
