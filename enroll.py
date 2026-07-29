#!/usr/bin/env python3
"""
Enroll your face so Lock on Absence can recognize you.

Captures multiple face samples, trains an LBPH model, and saves it.
Only the enrolled person will prevent the screen from locking.

Usage:
    python enroll.py                  # 30 samples (default)
    python enroll.py --samples 50     # 50 samples (more accurate)
    python enroll.py --camera 1       # use a different camera
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

# ────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll your face for Lock on Absence")
    parser.add_argument(
        "--samples", type=int, default=SAMPLES,
        help=f"Number of face samples to capture (default: {SAMPLES})",
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

    # ── Haar cascade ──
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        log("ERROR: failed to load Haar cascade")
        sys.exit(1)

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
    log("Center your face and slowly move your head:")
    log("  -> front, left, right, up, down")
    log("")

    faces_data: list = []
    labels: list = []
    count = 0
    multi_warn = 0  # throttle "multiple faces" warnings

    while count < args.samples:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(80, 80))

        if len(faces) == 1:
            x, y, w, h = faces[0]
            roi = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
            faces_data.append(roi)
            labels.append(1)
            count += 1
            multi_warn = 0

            bar = "#" * (count * 30 // args.samples)
            blank = " " * (30 - len(bar))
            print(f"\r  [{bar}{blank}] {count}/{args.samples}", end="", flush=True)

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
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces_data, np.array(labels))

    recognizer.write(args.output)
    log(f"Model saved to: {args.output}")
    log("")
    log("Enrollment complete! Run: python lock-on-absence.py")


if __name__ == "__main__":
    main()
