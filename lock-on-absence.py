#!/usr/bin/env python3
"""
Lock on Absence — Auto-lock your screen when you walk away.

Features:
  - Multi-angle face detection (frontal + profile)
  - Optional facial recognition (only YOU prevent the lock)
  - Keep-awake mode (prevents sleep/lock while you're present)

Setup:
  1. pip install -r requirements.txt
  2. (optional) python enroll.py            # train facial recognition
  3. python lock-on-absence.py              # start monitoring

Works on Windows, Linux, and macOS.
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ── Constants ──────────────────────────────────────────────────────
# Windows API for keep-awake
if sys.platform == "win32":
    _ES_CONTINUOUS = 0x80000000
    _ES_DISPLAY_REQUIRED = 0x00000002
    _ES_SYSTEM_REQUIRED = 0x00000001
    _SetThreadExecutionState = ctypes.windll.kernel32.SetThreadExecutionState
    _SetThreadExecutionState.argtypes = [ctypes.c_ulong]
    _SetThreadExecutionState.restype = ctypes.c_ulong
else:
    _ES_CONTINUOUS = _ES_DISPLAY_REQUIRED = _ES_SYSTEM_REQUIRED = 0
    _SetThreadExecutionState = None

# ── Defaults (tunable via CLI) ─────────────────────────────────────
CHECK_INTERVAL = 1.5
ABSENCE_SECONDS = 10
CAMERA_INDEX = 0
SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 3
MIN_FACE_SIZE = (40, 40)
FRAME_WIDTH = 640
POST_LOCK_COOLDOWN = 30
RECOGNITION_THRESHOLD = 75

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(SCRIPT_DIR / "face_model.yml")

# Multi-angle cascades (OpenCV built-in)
_CASCADE_DIR = cv2.data.haarcascades
CASCADE_PATHS = [
    _CASCADE_DIR + "haarcascade_frontalface_default.xml",   # front
    _CASCADE_DIR + "haarcascade_profileface.xml",            # side profile
]

# ────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── System control ─────────────────────────────────────────────────

def keep_awake() -> bool:
    """Prevent the system from sleeping/locking the display."""
    if sys.platform == "win32" and _SetThreadExecutionState:
        _SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_DISPLAY_REQUIRED | _ES_SYSTEM_REQUIRED
        )
        return True
    # Linux: use systemd-inhibit if available, otherwise xdg-screensaver
    try:
        subprocess.run(
            ["systemd-inhibit", "--what=idle:sleep", "--why=Lock on Absence",
             "--who=lock-on-absence", "true"],
            timeout=1, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        pass
    try:
        subprocess.run(
            ["xdg-screensaver", "reset"],
            timeout=1, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def allow_sleep() -> None:
    """Allow the system to sleep/lock normally again."""
    if sys.platform == "win32" and _SetThreadExecutionState:
        _SetThreadExecutionState(_ES_CONTINUOUS)


def lock_screen() -> bool:
    """Lock the workstation. Returns True on success."""
    if sys.platform == "win32":
        # Release keep-awake before locking so the lock sticks
        allow_sleep()
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


# ── Face detection ─────────────────────────────────────────────────

def load_cascades() -> list[cv2.CascadeClassifier]:
    """Load all available Haar cascades (frontal + profile)."""
    cascades = []
    for path in CASCADE_PATHS:
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                cascades.append(c)
    if not cascades:
        log("ERROR: no Haar cascades available")
        sys.exit(1)
    return cascades


def detect_faces_all(cascades: list, frame: np.ndarray) -> list:
    """
    Detect faces using ALL cascades (frontal + profile).
    Returns deduplicated list of rectangles [(x, y, w, h), ...].
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_faces = []
    for cascade in cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )
        all_faces.extend(faces)
    # Deduplicate overlapping rectangles (keep the larger one)
    if len(all_faces) <= 1:
        return all_faces
    return _dedup_faces(all_faces)


def _dedup_faces(faces: list) -> list:
    """Remove duplicate/overlapping face rectangles."""
    if not faces:
        return []
    # Sort by area descending
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for rect in faces:
        x, y, w, h = rect
        overlap = False
        for kx, ky, kw, kh in kept:
            # Check intersection over area
            xi = max(x, kx)
            yi = max(y, ky)
            wi = min(x + w, kx + kw) - xi
            hi = min(y + h, ky + kh) - yi
            if wi > 0 and hi > 0:
                inter = wi * hi
                area = w * h
                if inter > area * 0.5:
                    overlap = True
                    break
        if not overlap:
            kept.append(rect)
    return kept


# ── Recognition ────────────────────────────────────────────────────

def recognize_face(
    recognizer: cv2.face.LBPHFaceRecognizer,
    gray_frame: np.ndarray,
    face_rect: tuple,
) -> tuple[bool, float]:
    """
    Return (is_owner, confidence).
    confidence is LBPH distance — lower = better match.
    """
    x, y, w, h = face_rect
    roi = cv2.resize(gray_frame[y : y + h, x : x + w], (200, 200))
    _label, confidence = recognizer.predict(roi)
    return confidence < RECOGNITION_THRESHOLD, confidence


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lock screen when you leave — multi-angle face detection + keep-awake."
    )
    parser.add_argument(
        "--delay", type=int, default=ABSENCE_SECONDS,
        help=f"Seconds without owner before locking (default: {ABSENCE_SECONDS})",
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="Dry-run: detect but never lock",
    )
    parser.add_argument(
        "--check-interval", type=float, default=CHECK_INTERVAL,
        help=f"Seconds between checks (default: {CHECK_INTERVAL})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Path to trained LBPH model (default: ./face_model.yml)",
    )
    parser.add_argument(
        "--no-keep-awake", action="store_true",
        help="Disable keep-awake (allow normal sleep/lock behavior even when owner present)",
    )
    args = parser.parse_args()

    # ── Load cascades (frontal + profile) ──
    cascades = load_cascades()
    log(f"Loaded {len(cascades)} Haar cascade(s)")

    # ── Recognition model (optional) ──
    recognizer = None
    if os.path.exists(args.model):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(args.model)
        log(f"Face model loaded: {args.model}")
        log("Mode: OWNER RECOGNITION (only your face prevents lock)")
    else:
        log(f"No face model at {args.model} — run 'python enroll.py' first.")
        log("Mode: ANY FACE (any face prevents lock)")

    # ── Webcam ──
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        log(f"ERROR: camera {args.camera} not available")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(FRAME_WIDTH * 0.75))

    # Warmup
    for _ in range(8):
        cap.read()
        time.sleep(0.1)

    # ── Status ──
    keep_awake_enabled = not args.no_keep_awake
    lock_label = "DRY-RUN" if args.no_lock else "ACTIVE LOCK"
    awake_label = "keep-awake ON" if keep_awake_enabled else "keep-awake OFF"
    log(f"Monitoring — delay={args.delay}s  interval={args.check_interval}s  [{lock_label}]  [{awake_label}]")
    log("Press Ctrl+C to stop")

    absence_start: float | None = None
    locked = False
    was_awake = False  # track keep-awake state to avoid redundant calls

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(2)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces_all(cascades, frame)
            owner_present = False

            if len(faces) == 0:
                owner_present = False
            elif recognizer is not None:
                for rect in faces:
                    is_owner, _conf = recognize_face(recognizer, gray, rect)
                    if is_owner:
                        owner_present = True
                        break
            else:
                owner_present = True  # any face = owner

            if owner_present:
                # Owner present: keep screen awake, reset lock timer
                if keep_awake_enabled and not was_awake:
                    keep_awake()
                    was_awake = True

                if absence_start is not None:
                    log("Owner detected — timer reset")
                absence_start = None
                locked = False
            else:
                # No owner: allow sleep
                if was_awake:
                    allow_sleep()
                    was_awake = False

                if locked:
                    time.sleep(args.check_interval)
                    continue

                # INTRUDER: face detected but NOT the owner -> instant lock
                if len(faces) > 0 and recognizer is not None:
                    if args.no_lock:
                        log(">>> [DRY-RUN] Would lock NOW (intruder detected)")
                    else:
                        log(">>> LOCKING NOW (intruder detected)")
                        lock_screen()
                        locked = True
                        log(f"Cooldown {POST_LOCK_COOLDOWN}s...")
                    absence_start = None
                else:
                    # EMPTY: no face at all -> use normal delay
                    if absence_start is None:
                        absence_start = time.time()
                        log(f"No face — waiting {args.delay}s...")
                    elif time.time() - absence_start >= args.delay:
                        if args.no_lock:
                            log(f">>> [DRY-RUN] Would lock now (nobody present)")
                        else:
                            log(f">>> LOCKING (nobody present)")
                            lock_screen()
                            locked = True
                            log(f"Cooldown {POST_LOCK_COOLDOWN}s...")
                        absence_start = None

            time.sleep(args.check_interval)

    except KeyboardInterrupt:
        log("Stopped by user")
    finally:
        allow_sleep()
        cap.release()


if __name__ == "__main__":
    main()
