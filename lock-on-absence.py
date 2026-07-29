#!/usr/bin/env python3
"""
Lock on Absence — Auto-lock your screen when you walk away.

Dual mode:
  - Simple: locks when NO face is detected (any face keeps it unlocked)
  - Recognition: locks when the OWNER's face is NOT detected — a different
    person sitting down also triggers the lock.

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


# ── Defaults (tunable via CLI) ─────────────────────────────────────
CHECK_INTERVAL = 1.5       # seconds between checks
ABSENCE_SECONDS = 10       # seconds without owner before locking
CAMERA_INDEX = 0            # default webcam
SCALE_FACTOR = 1.05         # Haar cascade pyramid scale
MIN_NEIGHBORS = 3           # Haar cascade neighbours
MIN_FACE_SIZE = (40, 40)    # smallest detectable face
FRAME_WIDTH = 640           # capture resolution
POST_LOCK_COOLDOWN = 30     # seconds of silence after a lock

# LBPH recognition: confidence below this = owner match (lower = better)
RECOGNITION_THRESHOLD = 75

# Paths (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DEFAULT_MODEL = str(SCRIPT_DIR / "face_model.yml")

# ────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def lock_screen() -> bool:
    """Lock the workstation. Returns True on success."""
    if sys.platform == "win32":
        ctypes.windll.user32.LockWorkStation()
        return True

    # Linux / BSD
    for args in (
        ["loginctl", "lock-session"],
        ["xdg-screensaver", "lock"],
        ["gnome-screensaver-command", "--lock"],
        ["dm-tool", "lock"],
        ["i3lock", "-n"],
        ["slock"],
        ["osascript", "-e", "tell application \"System Events\" to sleep"],  # macOS
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


def detect_faces(cascade: cv2.CascadeClassifier, frame: np.ndarray) -> list:
    """Return list of face rectangles [(x, y, w, h), ...]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE,
    )


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
        description="Lock screen when you leave — with optional facial recognition."
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
    args = parser.parse_args()

    # ── Haar cascade ──
    if not os.path.exists(HAAR_PATH):
        log(f"ERROR: Haar cascade not found at {HAAR_PATH}")
        log("Reinstall opencv-python:  pip install --force-reinstall opencv-python")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    if face_cascade.empty():
        log("ERROR: failed to load Haar cascade")
        sys.exit(1)

    # ── Recognition model (optional) ──
    recognizer = None
    if os.path.exists(args.model):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(args.model)
        log(f"Face model loaded: {args.model}")
        log("Mode: OWNER RECOGNITION (only your face prevents lock)")
    else:
        log(f"No face model found at {args.model} — run 'python enroll.py' first.")
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

    lock_label = "DRY-RUN" if args.no_lock else "ACTIVE LOCK"
    log(f"Monitoring — delay={args.delay}s  interval={args.check_interval}s  [{lock_label}]")
    log("Press Ctrl+C to stop")

    absence_start: float | None = None
    locked = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(2)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(face_cascade, frame)
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
                if absence_start is not None:
                    log("Owner detected — timer reset")
                absence_start = None
                locked = False
            else:
                if locked:
                    time.sleep(args.check_interval)
                    continue

                if absence_start is None:
                    absence_start = time.time()
                    reason = "No face" if len(faces) == 0 else "Face NOT recognized"
                    log(f"{reason} — waiting {args.delay}s...")
                elif time.time() - absence_start >= args.delay:
                    reason = "nobody present" if len(faces) == 0 else "face not owner"
                    if args.no_lock:
                        log(f">>> [DRY-RUN] Would lock now ({reason})")
                    else:
                        log(f">>> LOCKING ({reason})")
                        lock_screen()
                        locked = True
                        log(f"Cooldown {POST_LOCK_COOLDOWN}s...")
                    absence_start = None

            time.sleep(args.check_interval)

    except KeyboardInterrupt:
        log("Stopped by user")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
