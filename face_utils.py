#!/usr/bin/env python3
"""Shared utilities for Lock on Absence — face detection, body detection, camera, logging."""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Logger
# ═══════════════════════════════════════════════════════════════════════

class Logger:
    """Timestamped logger with optional file output."""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath

    def __call__(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if self.filepath:
            try:
                with open(self.filepath, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════
#  Cascade loading
# ═══════════════════════════════════════════════════════════════════════

_CASCADE_DIR = cv2.data.haarcascades

CASCADE_PATHS = [
    _CASCADE_DIR + "haarcascade_frontalface_default.xml",
    _CASCADE_DIR + "haarcascade_profileface.xml",
]


def load_cascades(log: Logger | None = None) -> list:
    """Load all available Haar cascades (frontal + profile). Exits on failure."""
    cascades = []
    for path in CASCADE_PATHS:
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                cascades.append(c)
    if not cascades:
        msg = "ERROR: no Haar cascades available"
        if log:
            log(msg)
        else:
            print(msg, flush=True)
        sys.exit(1)
    return cascades


# ═══════════════════════════════════════════════════════════════════════
#  Multi-angle face detection
# ═══════════════════════════════════════════════════════════════════════

def detect_faces(
    cascades: list,
    frame: np.ndarray,
    scale_factor: float = 1.03,
    min_neighbors: int = 2,
    min_size: tuple = (30, 30),
) -> list:
    """
    Detect faces using all cascades (frontal + left/right profile via mirror).
    Returns deduplicated list of rectangles [(x, y, w, h), ...].
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    all_faces: list = []

    for cascade in cascades:
        # Normal detection (frontal + left profile)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        all_faces.extend(faces)

        # Mirror detection for right profile
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = cascade.detectMultiScale(
            gray_flipped,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        for (fx, fy, fw, fh) in faces_flipped:
            all_faces.append((w - fx - fw, fy, fw, fh))

    if len(all_faces) <= 1:
        return all_faces
    return _dedup_faces(all_faces)


def _dedup_faces(faces: list) -> list:
    """Remove duplicate/overlapping face rectangles. Keeps largest by area first."""
    if not faces:
        return []
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    kept: list = []
    for x, y, fw, fh in faces:
        overlap = False
        for kx, ky, kw, kh in kept:
            xi = max(x, kx)
            yi = max(y, ky)
            wi = min(x + fw, kx + kw) - xi
            hi = min(y + fh, ky + kh) - yi
            if wi > 0 and hi > 0 and (wi * hi) > (fw * fh) * 0.5:
                overlap = True
                break
        if not overlap:
            kept.append((x, y, fw, fh))
    return kept


# ═══════════════════════════════════════════════════════════════════════
#  Angle estimation (for enrollment UI)
# ═══════════════════════════════════════════════════════════════════════

def estimate_angle(x: int, w: int, frame_width: int) -> str:
    """Estimate head angle from face position in frame."""
    rel_x = (x + w / 2) / frame_width
    if rel_x < 0.35:
        return "LEFT"
    elif rel_x > 0.65:
        return "RIGHT"
    return "FRONT"


# ═══════════════════════════════════════════════════════════════════════
#  Body presence detection (frame differencing)
# ═══════════════════════════════════════════════════════════════════════

class BodyDetector:
    """
    Detects whether the owner's body is still in the chair via frame differencing.

    Compares current frame against a reference frame captured when the owner's
    face was last recognized. Auto-calibrates the difference threshold during
    the first seconds of operation to adapt to the specific camera and lighting.
    """

    def __init__(
        self,
        threshold: float = 18.0,
        ref_interval: float = 30.0,
        calibration_samples: int = 20,
        calibration_multiplier: float = 2.5,
    ):
        self.threshold = threshold
        self.ref_interval = ref_interval
        self.calibration_samples = calibration_samples
        self.calibration_multiplier = calibration_multiplier
        # Internal state
        self.ref_frame: np.ndarray | None = None
        self._last_ref_time = 0.0
        self._noise_samples: list[float] = []
        self._calibrated = False

    def update_ref(self, gray_frame: np.ndarray) -> None:
        """Update reference frame (call when owner face is detected)."""
        now = time.time()
        if now - self._last_ref_time >= self.ref_interval:
            self.ref_frame = gray_frame.copy()
            self._last_ref_time = now

    def present(self, current_frame: np.ndarray) -> bool:
        """
        Check whether the body is still present.
        Returns True if scene hasn't changed significantly vs reference.
        """
        if self.ref_frame is None:
            return False
        try:
            # Downscale for speed (~30x fewer pixels) and noise reduction
            small_ref = cv2.resize(self.ref_frame, (160, 120))
            small_cur = cv2.resize(current_frame, (160, 120))
            diff = cv2.absdiff(small_ref, small_cur)
            mean_diff = float(np.mean(diff))

            # Calibrate threshold from normal micro-movements
            if not self._calibrated and mean_diff > 0:
                self._noise_samples.append(mean_diff)
                if len(self._noise_samples) >= self.calibration_samples:
                    baseline = sum(self._noise_samples) / len(self._noise_samples)
                    # Departure = multiplier × baseline noise, with a floor
                    self.threshold = max(baseline * self.calibration_multiplier, 8.0)
                    self._calibrated = True

            return mean_diff < self.threshold
        except Exception:
            return False

    @property
    def status(self) -> str:
        """Human-readable threshold info."""
        if self._calibrated:
            return f"auto={self.threshold:.1f}"
        return f"calibrating... ({len(self._noise_samples)}/{self.calibration_samples})"


# ═══════════════════════════════════════════════════════════════════════
#  Camera
# ═══════════════════════════════════════════════════════════════════════

def open_camera(index: int = 0, width: int = 640) -> cv2.VideoCapture:
    """
    Open webcam with platform-appropriate backend and automatic fallback.
    Raises RuntimeError if no camera is available.
    """
    if sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 0.75))
            # Warmup — discard initial dark frames
            for _ in range(8):
                cap.read()
                time.sleep(0.05)
            return cap
        cap.release()

    raise RuntimeError(f"Camera index {index} not available")


def camera_available(index: int = 0) -> bool:
    """Quick check if camera is free (doesn't keep it open)."""
    if sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    ok = cap.isOpened()
    cap.release()
    return ok


class StealthCamera:
    """
    Camera wrapper that opens/closes per frame to minimize LED glow.
    Uses aggressive cleanup to force hardware release on each cycle.
    Trade-off: extra CPU per check, LED should blink instead of solid.
    """

    def __init__(self, index: int = 0, width: int = 640):
        self.index = index
        self.width = width
        self.height = int(width * 0.75)

    def read(self) -> tuple[bool, np.ndarray | None]:
        import gc
        # Use default backend — DSHOW sometimes holds the device after release
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            del cap
            return False, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Discard first frames (dark)
        for _ in range(3):
            cap.read()
            time.sleep(0.03)
        ret, frame = cap.read()
        # Aggressive release: del + gc to force hardware power-down
        cap.release()
        del cap
        gc.collect()
        return ret, frame

    def release(self) -> None:
        pass  # nothing to do; each read() opens and closes


# ═══════════════════════════════════════════════════════════════════════
#  Keep-awake (Windows + Linux)
# ═══════════════════════════════════════════════════════════════════════

if sys.platform == "win32":
    _ES_CONTINUOUS = 0x80000000
    _ES_DISPLAY_REQUIRED = 0x00000002
    _ES_SYSTEM_REQUIRED = 0x00000001
    import ctypes as _ctypes
    _SetThreadExecutionState = _ctypes.windll.kernel32.SetThreadExecutionState
    _SetThreadExecutionState.argtypes = [_ctypes.c_ulong]
    _SetThreadExecutionState.restype = _ctypes.c_ulong
else:
    _ES_CONTINUOUS = _ES_DISPLAY_REQUIRED = _ES_SYSTEM_REQUIRED = 0
    _SetThreadExecutionState = None


class KeepAwake:
    """
    Cross-platform keep-awake: prevents system sleep/lock while owner present.

    Windows: SetThreadExecutionState API.
    Linux:   systemd-inhibit subprocess (held alive, killed on release).
    macOS:   caffeinate subprocess (held alive, killed on release).
    """

    def __init__(self, log: Logger | None = None):
        self._active = False
        self._proc: subprocess.Popen | None = None
        self._log = log

    def enable(self) -> bool:
        """Start keeping the system awake. Returns True on success."""
        if self._active:
            return True  # already active

        if sys.platform == "win32" and _SetThreadExecutionState:
            _SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_DISPLAY_REQUIRED | _ES_SYSTEM_REQUIRED
            )
            self._active = True
            return True

        # Linux: systemd-inhibit with a long-lived sleep process
        try:
            self._proc = subprocess.Popen(
                ["systemd-inhibit", "--what=idle:sleep",
                 "--why=Lock on Absence: owner present",
                 "--who=lock-on-absence",
                 "sleep", "infinity"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._active = True
            return True
        except Exception:
            pass

        # macOS: caffeinate
        try:
            self._proc = subprocess.Popen(
                ["caffeinate", "-dimsu"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._active = True
            return True
        except Exception:
            pass

        return False

    def disable(self) -> None:
        """Release keep-awake. System resumes normal sleep/lock behavior."""
        if sys.platform == "win32" and _SetThreadExecutionState:
            _SetThreadExecutionState(_ES_CONTINUOUS)

        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

        self._active = False


# ═══════════════════════════════════════════════════════════════════════
#  Screen lock
# ═══════════════════════════════════════════════════════════════════════

def lock_screen(keep_awake: KeepAwake | None = None) -> bool:
    """
    Lock the workstation. Releases keep-awake first so the lock sticks.
    Returns True on success.
    """
    if keep_awake:
        keep_awake.disable()

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.LockWorkStation()
        return True

    for args in (
        ["loginctl", "lock-session"],
        ["xdg-screensaver", "lock"],
        ["gnome-screensaver-command", "--lock"],
        ["dm-tool", "lock"],
        ["i3lock", "-n"],
        ["slock"],
        ["osascript", "-e", 'tell application "System Events" to sleep'],
    ):
        try:
            subprocess.run(
                args, timeout=5, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            continue
    return False


# ═══════════════════════════════════════════════════════════════════════
#  Graceful shutdown
# ═══════════════════════════════════════════════════════════════════════

def install_signal_handlers(
    cap: cv2.VideoCapture,
    keep_awake: KeepAwake,
    log: Logger,
) -> None:
    """Install SIGINT/SIGTERM handlers for graceful cleanup."""

    def _cleanup(signum, frame):
        log(f"Received signal {signum} — shutting down")
        keep_awake.disable()
        cap.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
