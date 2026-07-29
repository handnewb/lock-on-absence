#!/usr/bin/env python3
"""
Enroll your face so Lock on Absence can recognize you — from any angle.

Captures face samples from multiple angles (frontal + profile) and
trains an LBPH model. Only the enrolled person will prevent screen lock.

Usage:
    python enroll.py                  # 30 samples (default)
    python enroll.py --samples 50     # 50 samples (more accurate)
    python enroll.py --camera 1       # use a different camera

Tips:
    Move your head: front -> left profile -> right profile -> up -> down.
    Good lighting and a clean background improve accuracy.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ── Defaults ───────────────────────────────────────────────────────
SAMPLES = 30
CAMERA_INDEX = 0
FRAME_WIDTH = 640
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = str(SCRIPT_DIR / "face_model.yml")

# Multi-angle cascades
_CASCADE_DIR = cv2.data.haarcascades
CASCADE_PATHS = [
    _CASCADE_DIR + "haarcascade_frontalface_default.xml",
    _CASCADE_DIR + "haarcascade_profileface.xml",
]

# ────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_cascades() -> list:
    """Load all available Haar cascades."""
    cascades = []
    for path in CASCADE_PATHS:
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                cascades.append((path.split("/")[-1].replace(".xml", ""), c))
    return cascades


def detect_any_face(cascades: list, gray: np.ndarray) -> list:
    """Detect faces using all cascades (including mirrored for right profile)."""
    h, w = gray.shape
    all_faces = []
    for _name, cascade in cascades:
        # Normal detection
        faces = cascade.detectMultiScale(gray, 1.03, 2, minSize=(60, 60))
        all_faces.extend(faces)
        # Mirrored for right profile
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = cascade.detectMultiScale(gray_flipped, 1.03, 2, minSize=(60, 60))
        for (fx, fy, fw, fh) in faces_flipped:
            all_faces.append((w - fx - fw, fy, fw, fh))
    if len(all_faces) <= 1:
        return all_faces
    # Dedup
    faces = sorted(all_faces, key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for rect in faces:
        x, y, w, h = rect
        overlap = False
        for kx, ky, kw, kh in kept:
            xi = max(x, kx)
            yi = max(y, ky)
            wi = min(x + w, kx + kw) - xi
            hi = min(y + h, ky + kh) - yi
            if wi > 0 and hi > 0 and (wi * hi) > (w * h) * 0.5:
                overlap = True
                break
        if not overlap:
            kept.append(rect)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll your face for Lock on Absence")
    parser.add_argument(
        "--samples", type=int, default=SAMPLES,
        help=f"Number of face samples (default: {SAMPLES})",
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output model path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    # ── Load cascades ──
    cascades = load_cascades()
    if not cascades:
        log("ERROR: no Haar cascades available")
        sys.exit(1)
    log(f"Loaded {len(cascades)} cascade(s): {', '.join(n for n, _ in cascades)}")

    # ── Webcam ──
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        log(f"ERROR: camera {args.camera} not available")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(FRAME_WIDTH * 0.75))

    # Warmup
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    log(f"Enrolling: {args.samples} samples needed")
    log("Move your head through all angles:")
    log("  -> FRONT -> LEFT profile -> RIGHT profile -> UP -> DOWN")
    log("")

    faces_data: list = []
    labels: list = []
    count = 0
    multi_warn = 0

    while count < args.samples:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_any_face(cascades, gray)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            roi = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
            faces_data.append(roi)
            labels.append(1)
            count += 1
            multi_warn = 0

            bar = "#" * (count * 30 // args.samples)
            blank = " " * (30 - len(bar))
            # Show which angle was detected (estimate from face position)
            frame_w = frame.shape[1]
            rel_x = (x + w / 2) / frame_w
            if rel_x < 0.35:
                angle = "LEFT"
            elif rel_x > 0.65:
                angle = "RIGHT"
            else:
                angle = "FRONT"
            print(f"\r  [{bar}{blank}] {count}/{args.samples} ({angle})", end="", flush=True)

        elif len(faces) > 1 and multi_warn % 15 == 0:
            log(f"\n  WARNING: {len(faces)} faces detected — keep only yourself in frame")

        multi_warn += 1
        time.sleep(0.12)

    print("")
    cap.release()
    log(f"Captured {len(faces_data)} samples")

    # ── Train ──
    log("Training LBPH model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8,
    )
    recognizer.train(faces_data, np.array(labels))

    recognizer.write(args.output)
    log(f"Model saved to: {args.output}")
    log("")
    log("Enrollment complete! Run: python lock-on-absence.py")


if __name__ == "__main__":
    main()
