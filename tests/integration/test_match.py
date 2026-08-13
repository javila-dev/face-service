import base64
import json

import cv2
import numpy as np
import pytest


def _jpeg_bytes(seed=3):
    img = np.random.default_rng(seed).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _one_hot(index: int, dim: int = 512):
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


def test_match_multipart_pass(client, fake_analyzer, make_face):
    emb = _one_hot(0)
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.95, embedding=emb)]
    resp = client.post(
        "/v1/match",
        headers={"X-API-Key": "test-key"},
        files={"image": ("live.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"embedding": json.dumps(emb.tolist()), "threshold": "0.5"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["match_passed"] is True
    assert body["match_score"] == pytest.approx(1.0, abs=1e-5)


def test_match_json_body_fail_different_person(client, fake_analyzer, make_face):
    live_emb = _one_hot(0)
    control_emb = _one_hot(1)
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.95, embedding=live_emb)]
    encoded = base64.b64encode(_jpeg_bytes()).decode("ascii")
    resp = client.post(
        "/v1/match",
        headers={"X-API-Key": "test-key"},
        json={"image": encoded, "embedding": control_emb.tolist(), "threshold": 0.4},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["match_passed"] is False
    assert body["match_score"] == pytest.approx(0.0, abs=1e-5)


def test_match_multiple_faces_hard_fails(client, fake_analyzer, make_face):
    fake_analyzer.next_faces = [
        make_face([10, 10, 100, 100]),
        make_face([200, 200, 300, 300]),
    ]
    resp = client.post(
        "/v1/match",
        headers={"X-API-Key": "test-key"},
        files={"image": ("live.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"embedding": json.dumps(_one_hot(0).tolist())},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["issues"][0]["code"] == "multiple_faces"


def test_match_embedding_dimension_mismatch_422(client, fake_analyzer, make_face):
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.95)]
    resp = client.post(
        "/v1/match",
        headers={"X-API-Key": "test-key"},
        files={"image": ("live.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"embedding": json.dumps([0.1, 0.2, 0.3])},
    )
    assert resp.status_code == 422


def test_match_missing_auth_401(client):
    resp = client.post(
        "/v1/match",
        files={"image": ("live.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"embedding": json.dumps(_one_hot(0).tolist())},
    )
    assert resp.status_code == 401
