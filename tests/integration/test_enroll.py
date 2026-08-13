import base64

import cv2
import numpy as np


def _jpeg_bytes(seed=7):
    img = np.random.default_rng(seed).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_enroll_multipart_success(client, fake_analyzer, make_face):
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.95)]
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert len(body["embedding"]) == 512
    assert body["issues"] == []


def test_enroll_json_base64_success(client, fake_analyzer, make_face):
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.95)]
    encoded = base64.b64encode(_jpeg_bytes()).decode("ascii")
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        json={"image": encoded},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_enroll_no_face_returns_ok_false_with_200(client, fake_analyzer):
    fake_analyzer.next_faces = []
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["issues"][0]["code"] == "no_face"
    assert body["embedding"] is None


def test_enroll_missing_api_key_401(client):
    resp = client.post(
        "/v1/enroll",
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 401


def test_enroll_invalid_image_400(client, fake_analyzer):
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", b"not-an-image", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_enroll_missing_image_field_400(client):
    resp = client.post("/v1/enroll", headers={"X-API-Key": "test-key"}, json={})
    assert resp.status_code == 400
