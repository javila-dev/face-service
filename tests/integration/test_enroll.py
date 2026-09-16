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


def test_enroll_missing_image_field_422(client):
    resp = client.post("/v1/enroll", headers={"X-API-Key": "test-key"}, json={})
    assert resp.status_code == 422


def test_enroll_multipart_max_yaw_blocks(client, fake_analyzer, make_face):
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.95, pose=[0.0, 40.0, 0.0])]
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"max_yaw": "10"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["issues"][0]["code"] == "bad_pose"


def test_enroll_json_max_yaw_blocks(client, fake_analyzer, make_face):
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.95, pose=[0.0, 40.0, 0.0])]
    encoded = base64.b64encode(_jpeg_bytes()).decode("ascii")
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        json={"image": encoded, "max_yaw": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["issues"][0]["code"] == "bad_pose"


def test_enroll_default_does_not_block_on_moderate_yaw(client, fake_analyzer, make_face):
    # sin max_yaw en el request, sigue siendo el comportamiento no-bloqueante de siempre
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.95, pose=[0.0, 40.0, 0.0])]
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["issues"][0]["code"] == "bad_pose"


def test_enroll_invalid_max_yaw_multipart_422(client, fake_analyzer):
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"max_yaw": "not-a-number"},
    )
    assert resp.status_code == 422


def test_enroll_max_yaw_out_of_range_json_422(client, fake_analyzer):
    encoded = base64.b64encode(_jpeg_bytes()).decode("ascii")
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        json={"image": encoded, "max_yaw": 999},
    )
    assert resp.status_code == 422


def test_enroll_multipart_min_det_score_override_blocks(client, fake_analyzer, make_face):
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.7)]
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"min_det_score": "0.9"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["issues"][0]["code"] == "low_confidence"


def test_enroll_json_min_det_score_override_blocks(client, fake_analyzer, make_face):
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.7)]
    encoded = base64.b64encode(_jpeg_bytes()).decode("ascii")
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        json={"image": encoded, "min_det_score": 0.9},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["issues"][0]["code"] == "low_confidence"


def test_enroll_default_does_not_block_on_moderate_det_score(client, fake_analyzer, make_face):
    # sin min_det_score en el request, sigue siendo el default de FACE_MIN_DET_SCORE (0.50)
    fake_analyzer.next_faces = [make_face([50, 50, 350, 350], det_score=0.7)]
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_enroll_min_det_score_out_of_range_json_422(client, fake_analyzer):
    encoded = base64.b64encode(_jpeg_bytes()).decode("ascii")
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        json={"image": encoded, "min_det_score": 2.0},
    )
    assert resp.status_code == 422


def test_enroll_min_resolution_out_of_range_multipart_422(client, fake_analyzer):
    resp = client.post(
        "/v1/enroll",
        headers={"X-API-Key": "test-key"},
        files={"image": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"min_resolution": "0"},
    )
    assert resp.status_code == 422
