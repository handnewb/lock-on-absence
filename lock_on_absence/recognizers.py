"""
Recognizer backends — LBPH (legacy) and SFace (embeddings).

WHY THIS FILE EXISTS, AND THE ONE BUG IT IS BUILT TO PREVENT
─────────────────────────────────────────────────────────────
The two backends score in OPPOSITE DIRECTIONS:

    LBPH   returns a chi-square DISTANCE.   Lower  = more similar. Accept if  d < 65
    SFace  returns a cosine SIMILARITY.     Higher = more similar. Accept if  s > 0.363

A single `if score < threshold` that survives a backend swap silently inverts the
security decision: every stranger becomes the owner, no error, no log, tests
green. That is precisely the class of failure this project has hit repeatedly, so
raw scores are never compared by callers here. Every backend returns a `Score`
that knows its own direction, and only `Score.accepted` decides.

`Recognizer.accepts()` is the ONLY place a comparison happens. If you add a third
backend, implement `direction` and you cannot get this wrong.
"""

from __future__ import annotations

import contextlib
import os
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .face_utils import file_sha256, safe_face_roi

# ── SFace model (OpenCV Zoo) ──────────────────────────────────────────────
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
# The Zoo stores models with git-lfs. raw.githubusercontent.com returns a
# ~130-byte LFS POINTER, not the model — downloading that and handing it to
# cv2 produces a confusing parse error. The media host serves the real bytes.
SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_URL_FALLBACK = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_MIN_BYTES = 1_000_000       # a pointer file is ~130 bytes; the model is ~37 MB

# Published operating points for SFace (OpenCV Zoo / SFace paper).
# Transferable across lighting in a way an LBPH distance is not — but
# "transferable" means you start closer, NOT that it is calibrated for your
# webcam. Measure with lock-on-absence-replay before trusting it.
SFACE_COSINE_THRESHOLD = 0.363
SFACE_L2_THRESHOLD = 1.128

LBPH_THRESHOLD = 65.0             # inherited, never measured. See MIGRATION.md


class Direction(str, Enum):
    """Which way is 'more similar'?"""
    HIGHER_IS_BETTER = "higher"   # similarity, e.g. cosine
    LOWER_IS_BETTER = "lower"     # distance, e.g. chi-square, L2


@dataclass(frozen=True)
class Score:
    """A comparison result that carries its own polarity."""
    value: float
    threshold: float
    direction: Direction

    @property
    def accepted(self) -> bool:
        if self.direction is Direction.HIGHER_IS_BETTER:
            return self.value > self.threshold
        return self.value < self.threshold

    @property
    def margin(self) -> float:
        """Signed distance from the decision boundary; positive means accepted."""
        if self.direction is Direction.HIGHER_IS_BETTER:
            return self.value - self.threshold
        return self.threshold - self.value

    def __str__(self) -> str:
        arrow = ">" if self.direction is Direction.HIGHER_IS_BETTER else "<"
        return (f"{self.value:.4f} {arrow} {self.threshold:.4f} "
                f"= {'accept' if self.accepted else 'reject'}")


@runtime_checkable
class Recognizer(Protocol):
    """Anything that can decide whether a detected face is an enrolled user."""
    name: str
    threshold: float
    direction: Direction

    def score(self, frame: np.ndarray, gray: np.ndarray, face) -> Score | None:
        """Score one detection, or None when the face cannot be evaluated."""
        ...


def accepts(rec: Recognizer, frame, gray, face) -> tuple[bool, Score | None]:
    """
    The single comparison site in the codebase.

    Callers get a bool and, for logging, the Score. Nobody outside this module
    compares a raw number to a threshold.
    """
    s = rec.score(frame, gray, face)
    if s is None:
        return False, None
    return s.accepted, s


def best_of(rec: Recognizer, frame, gray, faces) -> tuple[bool, object, Score | None]:
    """
    Evaluate every detection, return (owner_found, rect, best_score).

    Returns on the first acceptance. `best_score` is the closest to acceptance
    even when nothing matched, which is what makes --debug output useful for
    threshold tuning.
    """
    best: Score | None = None
    for face in faces:
        ok, s = accepts(rec, frame, gray, face)
        if s is not None and (best is None or s.margin > best.margin):
            best = s
        if ok:
            return True, face, s
    return False, None, best


# ═══════════════════════════════════════════════════════════════════════════
#  LBPH — legacy
# ═══════════════════════════════════════════════════════════════════════════

class LBPHRecognizer:
    """
    OpenCV LBPH. Kept as the default until SFace is measured on real video.

    Honest assessment: LBPH is a 2006 algorithm whose chi-square distances are
    scene-dependent, so its threshold does NOT transfer between lighting
    conditions. It also has no meaningful operating point in the literature —
    the 65 here is inherited from a different grid configuration.
    """

    name = "lbph"
    direction = Direction.LOWER_IS_BETTER

    def __init__(self, model_path: str | os.PathLike,
                 threshold: float = LBPH_THRESHOLD) -> None:
        self.model_path = str(model_path)
        self.threshold = float(threshold)
        self._rec = cv2.face.LBPHFaceRecognizer_create()
        self._rec.read(self.model_path)

    def score(self, frame, gray, face) -> Score | None:
        roi = safe_face_roi(gray, face)
        if roi is None:
            return None
        try:
            _label, distance = self._rec.predict(cv2.resize(roi, (200, 200)))
        except cv2.error:
            return None
        return Score(float(distance), self.threshold, self.direction)


# ═══════════════════════════════════════════════════════════════════════════
#  SFace — embeddings
# ═══════════════════════════════════════════════════════════════════════════

class SFaceRecognizer:
    """
    SFace embeddings with cosine similarity against enrolled templates.

    Requires the full YuNet detection row (15 columns: bbox + 5 landmarks), not
    just the bounding box: alignCrop uses the landmarks to normalise pose before
    embedding, and that alignment is most of why this generalises better than
    LBPH. A 4-tuple rect is rejected rather than silently mis-aligned.
    """

    name = "sface"
    direction = Direction.HIGHER_IS_BETTER

    def __init__(self, model_path: str | os.PathLike,
                 templates: np.ndarray,
                 threshold: float = SFACE_COSINE_THRESHOLD,
                 labels: list[str] | None = None) -> None:
        if templates is None or len(templates) == 0:
            raise ValueError("SFaceRecognizer needs at least one enrolled template")
        self.model_path = str(model_path)
        self.threshold = float(threshold)
        self.templates = np.asarray(templates, dtype=np.float32)
        self.labels = labels or [""] * len(self.templates)
        self._rec = cv2.FaceRecognizerSF.create(self.model_path, "")
        self.last_label: str = ""

    # ── embedding ─────────────────────────────────────────────────────────
    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray | None:
        """Align + embed one detection. Returns a 1x128 float32 feature."""
        row = np.asarray(face_row, dtype=np.float32).reshape(1, -1)
        if row.shape[1] < 15:
            raise ValueError(
                f"SFace needs the full YuNet row (15 columns), got {row.shape[1]}. "
                "Use YUNetDetector.detect_raw(); a plain (x,y,w,h) rect cannot be "
                "aligned and would silently degrade accuracy.")
        try:
            aligned = self._rec.alignCrop(frame, row)
            feat = self._rec.feature(aligned)
        except cv2.error:
            return None
        if feat is None or feat.size == 0:
            return None
        return np.asarray(feat, dtype=np.float32).reshape(1, -1)

    def score(self, frame, gray, face) -> Score | None:
        del gray                                    # SFace works on the colour frame
        feat = self.embed(frame, face)
        if feat is None:
            return None
        best, best_idx = -1.0, -1
        for idx, tmpl in enumerate(self.templates):
            sim = float(self._rec.match(
                feat, tmpl.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))
            if sim > best:
                best, best_idx = sim, idx
        self.last_label = self.labels[best_idx] if best_idx >= 0 else ""
        return Score(best, self.threshold, self.direction)


# ═══════════════════════════════════════════════════════════════════════════
#  Model download
# ═══════════════════════════════════════════════════════════════════════════

def download_sface(target_dir: str | os.PathLike = ".", log=print) -> str:
    """
    Fetch the SFace ONNX from the OpenCV Zoo.

    Guards against the git-lfs trap: raw.githubusercontent.com serves a ~130-byte
    pointer file for LFS objects. Downloading that and handing it to cv2 fails
    with an opaque ONNX parse error, and because the file then exists on disk,
    every later run reuses the broken copy. So: download atomically to .part,
    reject anything implausibly small, and only then rename.
    """
    dest = Path(target_dir) / SFACE_MODEL
    if dest.exists():
        if dest.stat().st_size >= SFACE_MIN_BYTES:
            return str(dest)
        log(f"WARNING: {dest.name} is only {dest.stat().st_size} bytes "
            f"(git-lfs pointer or truncated download) — refetching")
        dest.unlink()

    tmp = dest.with_suffix(".part")
    last_error: Exception | None = None
    for url in (SFACE_URL, SFACE_URL_FALLBACK):
        try:
            log(f"Downloading SFace model (~37 MB) from {url.split('/')[2]}...")
            urllib.request.urlretrieve(url, tmp)
            size = tmp.stat().st_size
            if size < SFACE_MIN_BYTES:
                raise OSError(
                    f"got {size} bytes — that is an LFS pointer, not the model")
            os.replace(tmp, dest)
            log(f"Saved {dest} ({size / 1_048_576:.1f} MB, "
                f"sha256 {file_sha256(dest)[:16]}...)")
            return str(dest)
        except Exception as exc:
            last_error = exc
            log(f"  failed: {exc}")
        finally:
            with contextlib.suppress(OSError):
                if tmp.exists():
                    tmp.unlink()
    raise RuntimeError(
        f"could not download {SFACE_MODEL}: {last_error}. "
        f"Download it manually from https://github.com/opencv/opencv_zoo "
        f"(models/face_recognition_sface/) and place it next to face_model.yml.")
