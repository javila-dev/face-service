import numpy as np

from app.config import settings
from app.schemas.common import IssueCode
from app.services import quality


def _uniform_image(h=400, w=400, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _random_image(h=400, w=400, seed=42):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_no_face_returns_no_face_issue():
    result = quality.analyze(_uniform_image(), faces=[])
    assert result.ok is False
    assert result.issues[0].code == IssueCode.NO_FACE


def test_multiple_faces_hard_fails(make_face):
    faces = [make_face([10, 10, 100, 100]), make_face([200, 200, 300, 300])]
    result = quality.analyze(_uniform_image(), faces=faces)
    assert result.ok is False
    assert result.issues[0].code == IssueCode.MULTIPLE_FACES


def test_low_resolution_short_circuits(make_face):
    img = _uniform_image(h=50, w=50)
    result = quality.analyze(img, faces=[make_face([0, 0, 40, 40])])
    assert result.ok is False
    assert result.issues[0].code == IssueCode.LOW_RESOLUTION


def test_face_too_small_flagged(make_face):
    img = _random_image(h=1000, w=1000)
    face = make_face([10, 10, 40, 40], det_score=0.95)
    result = quality.analyze(img, faces=[face])
    assert result.ok is False
    assert IssueCode.FACE_TOO_SMALL in {i.code for i in result.issues}


def test_low_confidence_flagged(make_face):
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.2)
    result = quality.analyze(img, faces=[face])
    assert result.ok is False
    assert IssueCode.LOW_CONFIDENCE in {i.code for i in result.issues}


def test_flat_bright_face_flagged_blurry_and_bright(make_face):
    img = _uniform_image(value=250)
    face = make_face([50, 50, 350, 350], det_score=0.95)
    result = quality.analyze(img, faces=[face])
    codes = {i.code for i in result.issues}
    assert IssueCode.BLURRY in codes
    assert IssueCode.TOO_BRIGHT in codes
    assert result.ok is False


def test_flat_dark_face_flagged_too_dark(make_face):
    img = _uniform_image(value=5)
    face = make_face([50, 50, 350, 350], det_score=0.95)
    result = quality.analyze(img, faces=[face])
    assert IssueCode.TOO_DARK in {i.code for i in result.issues}
    assert result.ok is False


def test_good_face_passes(make_face):
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.95)
    result = quality.analyze(img, faces=[face])
    assert result.ok is True
    assert result.issues == []
    assert result.face is face


def test_bad_pose_is_soft_and_does_not_block(make_face):
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.95, pose=[0.0, 60.0, 0.0])
    result = quality.analyze(img, faces=[face])
    assert IssueCode.BAD_POSE in {i.code for i in result.issues}
    assert result.ok is True


def test_bad_pose_blocks_when_globally_enabled(make_face, monkeypatch):
    monkeypatch.setattr(settings, "face_pose_blocking", True)
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.95, pose=[0.0, 60.0, 0.0])
    result = quality.analyze(img, faces=[face])
    assert IssueCode.BAD_POSE in {i.code for i in result.issues}
    assert result.ok is False


def test_bad_pose_blocks_when_request_supplies_max_yaw(make_face):
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.95, pose=[0.0, 40.0, 0.0])
    result = quality.analyze(img, faces=[face], max_yaw=20)
    assert IssueCode.BAD_POSE in {i.code for i in result.issues}
    assert result.ok is False


def test_bad_pose_request_override_can_be_looser_than_global(make_face, monkeypatch):
    monkeypatch.setattr(settings, "face_pose_blocking", True)
    img = _random_image()
    # el default global (35°) hubiera bloqueado yaw=40, pero este request pide 60°
    face = make_face([50, 50, 350, 350], det_score=0.95, pose=[0.0, 40.0, 0.0])
    result = quality.analyze(img, faces=[face], max_yaw=60)
    assert result.ok is True
    assert result.issues == []


def test_pitch_and_roll_disabled_by_default(make_face):
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.95, pose=[80.0, 0.0, 80.0])
    result = quality.analyze(img, faces=[face])
    assert result.ok is True
    assert result.issues == []


def test_max_pitch_blocks_when_requested(make_face):
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.95, pose=[50.0, 0.0, 0.0])
    result = quality.analyze(img, faces=[face], max_pitch=20)
    assert IssueCode.BAD_POSE in {i.code for i in result.issues}
    assert result.ok is False


def test_max_roll_blocks_when_requested(make_face):
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.95, pose=[0.0, 0.0, 50.0])
    result = quality.analyze(img, faces=[face], max_roll=20)
    assert IssueCode.BAD_POSE in {i.code for i in result.issues}
    assert result.ok is False


def test_min_det_score_override_can_be_stricter(make_face):
    img = _random_image()
    face = make_face([50, 50, 350, 350], det_score=0.7)
    result = quality.analyze(img, faces=[face], min_det_score=0.9)
    assert IssueCode.LOW_CONFIDENCE in {i.code for i in result.issues}
    assert result.ok is False


def test_min_det_score_override_can_be_looser_than_global(make_face):
    img = _random_image()
    # el default global (0.50) hubiera bloqueado det_score=0.3
    face = make_face([50, 50, 350, 350], det_score=0.3)
    result = quality.analyze(img, faces=[face], min_det_score=0.1)
    assert IssueCode.LOW_CONFIDENCE not in {i.code for i in result.issues}
    assert result.ok is True


def test_min_face_ratio_override_can_be_stricter(make_face):
    img = _random_image(h=1000, w=1000)
    # ratio ~10%: pasa el default global (5%) pero no un override más exigente (50%)
    face = make_face([100, 100, 416, 416], det_score=0.95)
    result = quality.analyze(img, faces=[face])
    assert IssueCode.FACE_TOO_SMALL not in {i.code for i in result.issues}

    result = quality.analyze(img, faces=[face], min_face_ratio=50)
    assert IssueCode.FACE_TOO_SMALL in {i.code for i in result.issues}
    assert result.ok is False


def test_min_laplacian_var_override_can_be_looser(make_face):
    img = _uniform_image(value=128)
    face = make_face([50, 50, 350, 350], det_score=0.95)
    result = quality.analyze(img, faces=[face], min_laplacian_var=0)
    assert IssueCode.BLURRY not in {i.code for i in result.issues}
    assert result.ok is True


def test_min_brightness_override_can_be_stricter(make_face):
    img = _uniform_image(value=100)
    face = make_face([50, 50, 350, 350], det_score=0.95)
    result = quality.analyze(img, faces=[face], min_brightness=150)
    assert IssueCode.TOO_DARK in {i.code for i in result.issues}
    assert result.ok is False


def test_max_brightness_override_can_be_stricter(make_face):
    img = _uniform_image(value=150)
    face = make_face([50, 50, 350, 350], det_score=0.95)
    result = quality.analyze(img, faces=[face], max_brightness=100)
    assert IssueCode.TOO_BRIGHT in {i.code for i in result.issues}
    assert result.ok is False


def test_min_resolution_override_can_be_stricter(make_face):
    img = _uniform_image(h=300, w=300)
    result = quality.analyze(img, faces=[make_face([10, 10, 250, 250])], min_resolution=500)
    assert result.ok is False
    assert result.issues[0].code == IssueCode.LOW_RESOLUTION


def test_min_resolution_override_can_be_looser_than_global(make_face):
    # el default global (200) hubiera bloqueado una imagen de 100x100
    img = _random_image(h=100, w=100)
    face = make_face([5, 5, 90, 90], det_score=0.95)
    result = quality.analyze(img, faces=[face], min_resolution=50)
    assert IssueCode.LOW_RESOLUTION not in {i.code for i in result.issues}
