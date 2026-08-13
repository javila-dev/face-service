import base64
from typing import List

import numpy as np

from app.core.errors import EmbeddingError
from app.schemas.common import IssueCode
from app.services.model_registry import EMBEDDING_DIM


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        raise EmbeddingError(IssueCode.INVALID_IMAGE, "El embedding extraído es degenerado (norma cero).")
    return vec / norm


def extract_embedding(face) -> np.ndarray:
    normed = getattr(face, "normed_embedding", None)
    if normed is not None:
        return np.asarray(normed, dtype=np.float32)
    return l2_normalize(np.asarray(face.embedding, dtype=np.float32))


def embedding_to_list(vec: np.ndarray) -> List[float]:
    return [float(x) for x in vec.tolist()]


def list_to_embedding(data) -> np.ndarray:
    if not isinstance(data, (list, tuple)) or not data:
        raise EmbeddingError(
            IssueCode.EMBEDDING_DIMENSION_MISMATCH, "El embedding recibido está vacío o no es una lista."
        )
    if len(data) != EMBEDDING_DIM:
        raise EmbeddingError(
            IssueCode.EMBEDDING_DIMENSION_MISMATCH,
            f"El embedding recibido tiene {len(data)} dimensiones, se esperaban {EMBEDDING_DIM}.",
        )
    try:
        return np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(
            IssueCode.EMBEDDING_DIMENSION_MISMATCH, "El embedding contiene valores no numéricos."
        ) from exc


def embedding_to_base64(vec: np.ndarray) -> str:
    return base64.b64encode(np.asarray(vec, dtype=np.float32).tobytes()).decode("ascii")


def base64_to_embedding(data: str) -> np.ndarray:
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise EmbeddingError(
            IssueCode.EMBEDDING_DIMENSION_MISMATCH, "El embedding en base64 no es válido."
        ) from exc
    vec = np.frombuffer(raw, dtype=np.float32)
    return list_to_embedding(vec.tolist())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))


def decide_match(score: float, threshold: float) -> bool:
    return score >= threshold
