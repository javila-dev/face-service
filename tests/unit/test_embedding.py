import numpy as np
import pytest

from app.core.errors import EmbeddingError
from app.services import embedding
from app.services.model_registry import EMBEDDING_DIM


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert embedding.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert embedding.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert embedding.cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-5)


def test_decide_match_boundary_is_inclusive():
    assert embedding.decide_match(0.4, 0.4) is True
    assert embedding.decide_match(0.399, 0.4) is False


def test_list_to_embedding_round_trip():
    data = [0.1] * EMBEDDING_DIM
    vec = embedding.list_to_embedding(data)
    back = embedding.embedding_to_list(vec)
    assert len(back) == EMBEDDING_DIM
    assert back[0] == pytest.approx(0.1, abs=1e-6)


def test_list_to_embedding_rejects_wrong_dimension():
    with pytest.raises(EmbeddingError):
        embedding.list_to_embedding([0.1, 0.2, 0.3])


def test_list_to_embedding_rejects_non_list():
    with pytest.raises(EmbeddingError):
        embedding.list_to_embedding("not-a-list")


def test_list_to_embedding_rejects_empty_list():
    with pytest.raises(EmbeddingError):
        embedding.list_to_embedding([])


def test_base64_embedding_round_trip():
    data = [0.25] * EMBEDDING_DIM
    vec = embedding.list_to_embedding(data)
    encoded = embedding.embedding_to_base64(vec)
    decoded = embedding.base64_to_embedding(encoded)
    assert np.allclose(decoded, vec, atol=1e-6)
