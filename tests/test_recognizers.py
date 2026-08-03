"""
Tests for the recognizer backends.

CONSTRAINT, STATED PLAINLY: the SFace ONNX (~37 MB) is stored in the OpenCV Zoo
with git-lfs, and the CI sandbox this was authored in could not reach the media
host that serves LFS content. So the SFace paths below are tested against a mock
that mimics cv2.FaceRecognizerSF's three methods (alignCrop / feature / match)
against the real signatures verified from cv2 5.0.0.

What that means honestly:
  * the abstraction, the score-direction logic, the alignment guard, the download
    guard and the harness wiring ARE tested here;
  * the numerical behaviour of the real network is NOT. `lock-on-absence-enroll
    --recognizer sface --self-test` exists to close that gap on a real machine in
    about 30 seconds, and it is the first thing to run after installing.

Tests that need the real model are marked `needs_sface` and skip when absent, so
they turn themselves on for anyone who has it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lock_on_absence.recognizers import (
    LBPH_THRESHOLD,
    SFACE_COSINE_THRESHOLD,
    SFACE_MIN_BYTES,
    SFACE_MODEL,
    Direction,
    LBPHRecognizer,
    Score,
    SFaceRecognizer,
    accepts,
    best_of,
    download_sface,
)

ROOT = Path(__file__).resolve().parent.parent
SFACE_PATH = ROOT / SFACE_MODEL
needs_sface = pytest.mark.skipif(
    not (SFACE_PATH.exists() and SFACE_PATH.stat().st_size >= SFACE_MIN_BYTES),
    reason=f"{SFACE_MODEL} not present (git-lfs; run enroll --recognizer sface once)")


# ═══════════════════════════════════════════════════════════════════════════
#  Score direction — the bug this module exists to prevent
# ═══════════════════════════════════════════════════════════════════════════

def test_lower_is_better_accepts_small_values():
    assert Score(40.0, 65.0, Direction.LOWER_IS_BETTER).accepted is True
    assert Score(80.0, 65.0, Direction.LOWER_IS_BETTER).accepted is False


def test_higher_is_better_accepts_large_values():
    assert Score(0.50, 0.363, Direction.HIGHER_IS_BETTER).accepted is True
    assert Score(0.20, 0.363, Direction.HIGHER_IS_BETTER).accepted is False


def test_the_two_backends_score_in_opposite_directions():
    """
    If this ever fails, a backend swap has silently inverted the security
    decision and every stranger is being accepted as the owner.
    """
    assert LBPHRecognizer.direction is Direction.LOWER_IS_BETTER
    assert SFaceRecognizer.direction is Direction.HIGHER_IS_BETTER
    assert LBPHRecognizer.direction is not SFaceRecognizer.direction


def test_same_number_means_opposite_verdicts_across_directions():
    """A naive `score < threshold` shared by both backends would invert here."""
    lo = Score(0.20, 0.363, Direction.LOWER_IS_BETTER)
    hi = Score(0.20, 0.363, Direction.HIGHER_IS_BETTER)
    assert lo.accepted is True and hi.accepted is False


def test_margin_sign_tracks_acceptance():
    for d in Direction:
        for v in (0.1, 0.5, 40.0, 90.0):
            s = Score(v, 0.363 if d is Direction.HIGHER_IS_BETTER else 65.0, d)
            assert (s.margin > 0) == s.accepted


# ═══════════════════════════════════════════════════════════════════════════
#  LBPH
# ═══════════════════════════════════════════════════════════════════════════

def _train_lbph(tmp_path: Path, seed: int = 0) -> Path:
    import cv2
    rng = np.random.default_rng(seed)
    rec = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=6, grid_y=6)
    rec.train([rng.integers(0, 255, (200, 200), dtype=np.uint8) for _ in range(5)],
              np.array([1] * 5))
    p = tmp_path / "face_model.yml"
    rec.write(str(p))
    return p


def test_lbph_scores_and_reports_direction(tmp_path):
    gray = np.random.default_rng(0).integers(0, 255, (480, 640), dtype=np.uint8)
    rec = LBPHRecognizer(_train_lbph(tmp_path))
    s = rec.score(None, gray, (100, 100, 200, 200))
    assert s is not None
    assert s.direction is Direction.LOWER_IS_BETTER
    assert s.threshold == LBPH_THRESHOLD
    assert s.value >= 0.0


def test_lbph_returns_none_for_unusable_roi(tmp_path):
    gray = np.full((480, 640), 128, np.uint8)
    rec = LBPHRecognizer(_train_lbph(tmp_path))
    assert rec.score(None, gray, (10_000, 10_000, 50, 50)) is None   # off-frame
    assert rec.score(None, gray, (0, 0, 5, 5)) is None               # too small


# ═══════════════════════════════════════════════════════════════════════════
#  SFace — against a mock of the verified cv2 API
# ═══════════════════════════════════════════════════════════════════════════

class _MockSF:
    """
    Mimics cv2.FaceRecognizerSF. Signatures verified against cv2 5.0.0:
        alignCrop(src_img, face_box) -> aligned_img
        feature(aligned_img)         -> 1x128 float32
        match(f1, f2, dis_type)      -> float
    """

    def __init__(self, feature_for=None):
        self._feature_for = feature_for or (lambda img: np.ones((1, 128), np.float32))
        self.align_calls: list[np.ndarray] = []

    def alignCrop(self, src_img, face_box):          # mirrors the cv2 name
        self.align_calls.append(np.asarray(face_box))
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, aligned_img):
        return self._feature_for(aligned_img)

    def match(self, f1, f2, dis_type=None):
        del dis_type
        a, b = np.asarray(f1).ravel(), np.asarray(f2).ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def _sface(monkeypatch, tmp_path, templates, mock=None, threshold=None, labels=None):
    import cv2
    model = tmp_path / SFACE_MODEL
    model.write_bytes(b"x" * 16)
    monkeypatch.setattr(cv2.FaceRecognizerSF, "create",
                        staticmethod(lambda *a, **k: mock or _MockSF()))
    return SFaceRecognizer(model, templates,
                           threshold=threshold or SFACE_COSINE_THRESHOLD,
                           labels=labels)


def _row(x=100, y=100, w=120, h=120):
    """A full 15-column YuNet row."""
    return np.array([x, y, w, h,
                     x + 40, y + 45, x + 80, y + 45,      # eyes
                     x + 60, y + 70,                      # nose
                     x + 45, y + 95, x + 75, y + 95,      # mouth
                     0.99], dtype=np.float32)


def test_sface_accepts_matching_template(monkeypatch, tmp_path):
    tmpl = np.ones((1, 128), np.float32)
    rec = _sface(monkeypatch, tmp_path, tmpl)
    s = rec.score(np.zeros((480, 640, 3), np.uint8), None, _row())
    assert s is not None and s.direction is Direction.HIGHER_IS_BETTER
    assert s.value == pytest.approx(1.0, abs=1e-5)
    assert s.accepted is True


def test_sface_rejects_orthogonal_template(monkeypatch, tmp_path):
    tmpl = np.zeros((1, 128), np.float32)
    tmpl[0, ::2] = 1.0
    other = np.zeros((1, 128), np.float32)
    other[0, 1::2] = 1.0                        # orthogonal to the query
    rec = _sface(monkeypatch, tmp_path, other,
                 mock=_MockSF(feature_for=lambda img: tmpl))
    s = rec.score(np.zeros((480, 640, 3), np.uint8), None, _row())
    assert s.value == pytest.approx(0.0, abs=1e-6)
    assert s.accepted is False


def test_sface_picks_the_best_of_several_templates(monkeypatch, tmp_path):
    q = np.zeros((1, 128), np.float32)
    q[0, :64] = 1.0
    far = np.zeros((1, 128), np.float32)
    far[0, 64:] = 1.0                       # orthogonal to q -> cosine 0
    near = q.copy()
    rec = _sface(monkeypatch, tmp_path, np.vstack([far, near]),
                 mock=_MockSF(feature_for=lambda img: q),
                 labels=["stranger", "owner"])
    s = rec.score(np.zeros((480, 640, 3), np.uint8), None, _row())
    assert s.value == pytest.approx(1.0, abs=1e-5)
    assert rec.last_label == "owner"


def test_sface_refuses_a_bare_rect_instead_of_misaligning(monkeypatch, tmp_path):
    """
    A 4-tuple would still 'work' through alignCrop and quietly lose the pose
    normalisation that is most of SFace's advantage. Loud failure beats silent
    degradation.
    """
    rec = _sface(monkeypatch, tmp_path, np.ones((1, 128), np.float32))
    with pytest.raises(ValueError, match="15 columns"):
        rec.score(np.zeros((480, 640, 3), np.uint8), None, (10, 10, 50, 50))


def test_sface_passes_landmarks_through_to_aligncrop(monkeypatch, tmp_path):
    mock = _MockSF()
    rec = _sface(monkeypatch, tmp_path, np.ones((1, 128), np.float32), mock=mock)
    rec.score(np.zeros((480, 640, 3), np.uint8), None, _row())
    assert mock.align_calls, "alignCrop was never called"
    assert mock.align_calls[0].reshape(1, -1).shape[1] >= 15, "landmarks were dropped"


def test_sface_requires_at_least_one_template(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        _sface(monkeypatch, tmp_path, np.empty((0, 128), np.float32))


def test_sface_returns_none_when_embedding_fails(monkeypatch, tmp_path):
    rec = _sface(monkeypatch, tmp_path, np.ones((1, 128), np.float32),
                 mock=_MockSF(feature_for=lambda img: np.empty((0, 0), np.float32)))
    assert rec.score(np.zeros((480, 640, 3), np.uint8), None, _row()) is None


# ═══════════════════════════════════════════════════════════════════════════
#  best_of / accepts
# ═══════════════════════════════════════════════════════════════════════════

class _FakeRec:
    name = "fake"
    direction = Direction.HIGHER_IS_BETTER
    threshold = 0.5

    def __init__(self, values):
        self.values = list(values)

    def score(self, frame, gray, face):
        v = self.values[int(face)]
        return None if v is None else Score(v, self.threshold, self.direction)


def test_best_of_returns_first_acceptance():
    rec = _FakeRec([0.1, 0.9, 0.95])
    ok, face, s = best_of(rec, None, None, [0, 1, 2])
    assert ok is True and face == 1 and s.value == 0.9


def test_best_of_reports_closest_miss_for_tuning():
    rec = _FakeRec([0.1, 0.45, 0.2])
    ok, face, s = best_of(rec, None, None, [0, 1, 2])
    assert ok is False and face is None
    assert s.value == 0.45, "must surface the near-miss so --debug can guide tuning"


def test_best_of_tolerates_unscoreable_faces():
    rec = _FakeRec([None, None, 0.8])
    ok, _face, s = best_of(rec, None, None, [0, 1, 2])
    assert ok is True and s.value == 0.8


def test_accepts_returns_none_score_when_unscoreable():
    ok, s = accepts(_FakeRec([None]), None, None, 0)
    assert ok is False and s is None


# ═══════════════════════════════════════════════════════════════════════════
#  Download guard — the git-lfs pointer trap
# ═══════════════════════════════════════════════════════════════════════════

def test_download_rejects_an_lfs_pointer_masquerading_as_the_model(tmp_path, capsys, monkeypatch):
    """
    raw.githubusercontent.com serves a ~130-byte pointer for LFS objects. Left on
    disk it is reused forever and cv2 fails with an opaque ONNX parse error.
    """
    monkeypatch.setattr("urllib.request.urlretrieve",
                        lambda url, dest: (_ for _ in ()).throw(OSError("network blocked")))
    pointer = tmp_path / SFACE_MODEL
    pointer.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        b"size 38400000\n")
    assert pointer.stat().st_size < SFACE_MIN_BYTES

    with pytest.raises(RuntimeError, match="could not download"):
        download_sface(tmp_path, log=lambda m: None)
    assert not pointer.exists(), "the bogus file must be removed, not reused"


def test_download_returns_existing_plausible_model(tmp_path):
    big = tmp_path / SFACE_MODEL
    big.write_bytes(b"\0" * (SFACE_MIN_BYTES + 1))
    assert download_sface(tmp_path, log=lambda m: None) == str(big)


def test_download_leaves_no_part_file_behind(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlretrieve",
                        lambda url, dest: (_ for _ in ()).throw(OSError("network blocked")))
    with pytest.raises(RuntimeError):
        download_sface(tmp_path, log=lambda m: None)
    assert not list(tmp_path.glob("*.part"))


# ═══════════════════════════════════════════════════════════════════════════
#  Real model — self-enabling
# ═══════════════════════════════════════════════════════════════════════════

@needs_sface
def test_real_sface_produces_a_128d_embedding():
    import cv2
    rec = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    feat = rec.feature(np.zeros((112, 112, 3), np.uint8))
    assert np.asarray(feat).reshape(1, -1).shape[1] == 128


@needs_sface
def test_real_sface_self_match_is_near_one():
    import cv2
    rec = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    f = rec.feature(img)
    assert rec.match(f, f, cv2.FaceRecognizerSF_FR_COSINE) == pytest.approx(1.0, abs=1e-4)


@needs_sface
def test_real_sface_threshold_sits_inside_the_cosine_range():
    assert -1.0 < SFACE_COSINE_THRESHOLD < 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  CLI wiring
# ═══════════════════════════════════════════════════════════════════════════

def test_enroll_exposes_the_new_flags():
    import subprocess
    import sys as _sys
    r = subprocess.run([_sys.executable, "-m", "lock_on_absence.enroll", "--help"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "--recognizer" in r.stdout
    assert "--self-test" in r.stdout


def test_self_test_exits_nonzero_when_the_model_is_absent(tmp_path):
    """
    A self-test that cannot fail is useless. Without the ONNX it must report a
    clear manual-download path and exit non-zero so CI and scripts can react.
    """
    import os as _os
    import subprocess
    import sys as _sys
    env = {**_os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([_sys.executable, "-m", "lock_on_absence.enroll", "--self-test"],
                       capture_output=True, text=True, timeout=180,
                       cwd=tmp_path, env=env)
    out = r.stdout + r.stderr
    if "[ok] model present" in out:
        pytest.skip("the real SFace model is available here")
    assert r.returncode == 1, f"expected exit 1, got {r.returncode}\n{out}"
    assert "opencv_zoo" in out, "must tell the user where to get the model"


def test_agent_refuses_sface_without_templates(tmp_path):
    import os as _os
    import subprocess
    import sys as _sys
    env = {**_os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [_sys.executable, "-m", "lock_on_absence.agent", "--recognizer", "sface"],
        capture_output=True, text=True, timeout=180, cwd=tmp_path, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 2, f"expected exit 2, got {r.returncode}\n{out}"
    assert "face_model_sface.npz" in out
    assert "--recognizer sface" in out, "must tell the user how to fix it"


def test_agent_exposes_recognizer_flag():
    import subprocess
    import sys as _sys
    r = subprocess.run([_sys.executable, "-m", "lock_on_absence.agent", "--help"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0
    assert "--recognizer" in r.stdout
    assert "sface" in r.stdout


def test_sface_threshold_range_is_enforced_per_backend(tmp_path, monkeypatch):
    """
    LBPH lives in [20,100]; cosine lives in [0,1]. A 65 leaking into the sface
    path would accept everything, since cosine maxes out at 1.0.
    """
    from lock_on_absence.recognizers import (
        LBPH_THRESHOLD,
        SFACE_COSINE_THRESHOLD,
    )
    assert 20.0 <= LBPH_THRESHOLD <= 100.0
    assert 0.0 < SFACE_COSINE_THRESHOLD < 1.0
    assert SFACE_COSINE_THRESHOLD < 20.0, (
        "the two ranges must not overlap, otherwise a stale threshold in "
        "face_model.json silently transfers between backends")


# ═══════════════════════════════════════════════════════════════════════════
#  SFace is the default  (v5.2)
# ═══════════════════════════════════════════════════════════════════════════

def test_enroll_defaults_to_sface():
    """
    The published cosine operating point beats an unmeasured LBPH distance, so
    new enrollments get SFace without anyone having to know to ask for it.

    Asserted against the parser default, not the help text: help wording is
    cosmetic and would make this test fail on a reword.
    """
    import argparse
    from unittest.mock import patch

    captured: dict = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        for action in self._actions:
            if action.dest == "recognizer":
                captured["default"] = action.default
        raise SystemExit(0)          # stop before the camera is touched

    from lock_on_absence import enroll as enroll_mod
    with patch.object(argparse.ArgumentParser, "parse_args", spy), \
            pytest.raises(SystemExit):
        enroll_mod.main()
    del real_parse
    assert captured.get("default") == "sface", (
        f"enroll should default to sface, got {captured.get('default')!r}")


def test_existing_lbph_enrollment_keeps_working(tmp_path, monkeypatch):
    """
    Backward compatibility is not optional: anyone who enrolled before v5.2 has
    metadata saying lbph, and flipping the default must not lock them out.
    """
    import cv2
    import numpy as _np

    from lock_on_absence.face_utils import file_sha256

    model = tmp_path / "face_model.yml"
    rng = _np.random.default_rng(3)
    rec = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=6, grid_y=6)
    rec.train([rng.integers(0, 255, (200, 200), dtype=_np.uint8) for _ in range(4)],
              _np.array([1] * 4))
    rec.write(str(model))
    (tmp_path / "face_model.json").write_text(json.dumps({
        "recognizer": "lbph", "threshold": 65,
        "model_sha256": file_sha256(model), "users": {"1": "legacy"},
    }))

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.agent",
         "--model", str(model), "--camera", "99"],
        capture_output=True, text=True, timeout=180, cwd=tmp_path, env=env)
    out = r.stdout + r.stderr
    assert "Recognizer: LBPH" in out, out
    assert r.returncode != 2 and r.returncode != 3, out


def test_sface_is_preferred_when_templates_exist(tmp_path):
    """No metadata, both artefacts present -> pick SFace, not the legacy backend."""
    import numpy as _np
    (tmp_path / "face_model.yml").write_text("dummy")
    _np.savez(tmp_path / "face_model_sface.npz",
              templates=_np.ones((1, 128), _np.float32), labels=_np.array(["me"]))

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.agent",
         "--model", str(tmp_path / "face_model.yml"), "--camera", "99"],
        capture_output=True, text=True, timeout=180, cwd=tmp_path, env=env)
    out = r.stdout + r.stderr
    # It will fail later (no ONNX, no camera) but must have chosen sface first.
    assert "LBPH" not in out or "SFace" in out, out
