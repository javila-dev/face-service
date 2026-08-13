import base64

import cv2
import numpy as np
import pytest

from app.core.errors import InvalidImageError
from app.services import image_input


def _fake_jpeg_bytes() -> bytes:
    img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_decode_from_bytes_valid_image():
    img = image_input.decode_from_bytes(_fake_jpeg_bytes())
    assert img.shape[2] == 3


def test_decode_from_bytes_garbage_raises():
    with pytest.raises(InvalidImageError):
        image_input.decode_from_bytes(b"not-an-image")


def test_bytes_from_base64_plain():
    raw = _fake_jpeg_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    assert image_input.bytes_from_base64_or_data_url(encoded) == raw


def test_bytes_from_base64_data_url_prefix():
    raw = _fake_jpeg_bytes()
    encoded = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    assert image_input.bytes_from_base64_or_data_url(encoded) == raw


def test_bytes_from_base64_invalid_raises():
    with pytest.raises(InvalidImageError):
        image_input.bytes_from_base64_or_data_url("not-valid-base64!!!")


def test_bytes_from_base64_empty_raises():
    with pytest.raises(InvalidImageError):
        image_input.bytes_from_base64_or_data_url("")


def test_size_cap_rejects_oversized_payload(monkeypatch):
    monkeypatch.setattr(image_input.settings, "face_max_image_mb", 0.0001)
    encoded = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")
    with pytest.raises(InvalidImageError):
        image_input.bytes_from_base64_or_data_url(encoded)
