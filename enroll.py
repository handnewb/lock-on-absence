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
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_utils import (
    Logger,
    detect_faces,
    estimate_angle,
    load_cascades,
    open_camera,
)

# ── Defaults ───────────────────────────────────────────────────────
SAMPLES = 30
CAMERA_INDEX = 0
FRAME_WIDTH = 640
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = str(SCRIPT_DIR / "face_model.yml")


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

    log = Logger()

    # ── Load cascades ──
    cascades = load_cascades(log)
    log(f"Loaded {len(cascades)} cascade(s)")

    # ── Webcam ──
    try:
        cap = open_camera(args.camera, FRAME_WIDTH)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)

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
        faces = detect_faces(cascades, frame, scale_factor=1.03, min_neighbors=2, min_size=(60, 60))

        if len(faces) == 1:
            x, y, w, h = faces[0]
            roi = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
            faces_data.append(roi)
            labels.append(1)
            count += 1
            multi_warn = 0

            bar = "#" * (count * 30 // args.samples)
            blank = " " * (30 - len(bar))
            angle = estimate_angle(x, w, frame.shape[1])
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

    # ── Auto-calibrate recognition threshold ──
    log("Calibrating recognition threshold...")
    confidences = []
    for sample in faces_data:
        _label, conf = recognizer.predict(sample)
        confidences.append(conf)
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))
    # Threshold = mean + 2.5σ, clamped between 30 and 95
    calibrated = max(min(mean_conf + 2.5 * std_conf, 95.0), 30.0)
    log(f"  Mean confidence: {mean_conf:.1f}")
    log(f"  Std deviation:   {std_conf:.1f}")
    log(f"  Calibrated threshold: {calibrated:.0f} (lower = stricter, default was 85)")

    # Save threshold alongside model
    try:
        meta_path = str(Path(args.output).with_suffix(".json"))
        with open(meta_path, "w") as f:
            json.dump({
                "threshold": round(calibrated, 1),
                "mean_confidence": round(mean_conf, 1),
                "std_confidence": round(std_conf, 1),
                "samples": len(faces_data),
            }, f, indent=2)
        log(f"Threshold saved to: {meta_path}")
    except OSError as e:
        log(f"WARNING: Could not save threshold metadata: {e}")
        log("Recognition will use default threshold (85)")

    log("")
    log("Enrollment complete! Run: python lock-on-absence.py")


if __name__ == "__main__":
    main()
