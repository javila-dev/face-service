import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.services import model_registry


class FakeFace:
    """Stand-in for an insightface Face object, fully controllable in tests."""

    def __init__(self, bbox, det_score: float = 0.95, embedding=None, pose=None):
        self.bbox = bbox
        self.det_score = det_score
        self.pose = pose
        if embedding is None:
            rng = np.random.default_rng(abs(hash(tuple(bbox))) % (2**32))
            vec = rng.random(512).astype(np.float32)
            embedding = vec / np.linalg.norm(vec)
        self.normed_embedding = np.asarray(embedding, dtype=np.float32)


class FakeAnalyzer:
    """Stand-in for insightface.app.FaceAnalysis — never loads a real model."""

    def __init__(self):
        self.next_faces: list = []

    def get(self, img):
        return self.next_faces


@pytest.fixture
def make_face():
    return FakeFace


@pytest.fixture(autouse=True)
def _reset_model_registry():
    model_registry.reset_for_testing(None, ready=False)
    yield
    model_registry.reset_for_testing(None, ready=False)


@pytest.fixture(autouse=True)
def _configure_api_keys():
    original = settings.face_api_keys
    settings.face_api_keys = "test-key"
    yield
    settings.face_api_keys = original


@pytest.fixture
def fake_analyzer():
    analyzer = FakeAnalyzer()
    model_registry.reset_for_testing(analyzer, ready=True)
    return analyzer


@pytest.fixture
def client(fake_analyzer, monkeypatch):
    monkeypatch.setattr(model_registry, "load_model", lambda: None)
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_not_ready(monkeypatch):
    monkeypatch.setattr(model_registry, "load_model", lambda: None)
    from app.main import app

    with TestClient(app) as c:
        yield c
