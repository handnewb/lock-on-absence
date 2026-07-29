#!/usr/bin/env python3
"""
Enroll one or more faces so Lock on Absence can recognize authorized users.

Captures face samples from multiple angles (frontal + profile) and
trains an LBPH model. All enrolled users will prevent screen lock.

Usage:
    python enroll.py                           # single user, 30 samples
    python enroll.py --samples 50              # 50 samples (more accurate)
    python enroll.py --users Everton,Ana       # two authorized users
    python enroll.py --camera 1                # use a different camera

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


def _enroll_user(log: Logger, cap: cv2.VideoCapture, cascades: list,
                 user_label: int, user_name: str, samples_needed: int) -> list:
    """Capture face samples for one user. Returns list of (roi, label) tuples."""
    log(f"")
    log(f"=== User {user_label}: {user_name} ===")
    log(f"Move your head through all angles:")
    log(f"  -> FRONT -> LEFT profile -> RIGHT profile -> UP -> DOWN")
    log(f"")

    data: list = []
    count = 0
    multi_warn = 0

    while count < samples_needed:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(cascades, frame, scale_factor=1.03, min_neighbors=2, min_size=(60, 60))

        if len(faces) == 1:
            x, y, w, h = faces[0]
            roi = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
            data.append((roi, user_label))
            count += 1
            multi_warn = 0

            bar = "#" * (count * 30 // samples_needed)
            blank = " " * (30 - len(bar))
            angle = estimate_angle(x, w, frame.shape[1])
            print(f"\r  [{bar}{blank}] {count}/{samples_needed} ({angle})", end="", flush=True)

        elif len(faces) > 1 and multi_warn % 15 == 0:
            log(f"\n  WARNING: {len(faces)} faces detected — keep only yourself in frame")

        multi_warn += 1
        time.sleep(0.12)

    print("")
    log(f"Captured {len(data)} samples for {user_name}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll faces for Lock on Absence")
    parser.add_argument(
        "--samples", type=int, default=SAMPLES,
        help=f"Number of face samples per user (default: {SAMPLES})",
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output model path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--users", type=str, default=None,
        help="Comma-separated user names for multi-user enrollment (e.g. 'Everton,Ana')",
    )
    args = parser.parse_args()

    log = Logger()

    # Parse users
    if args.users:
        user_names = [n.strip() for n in args.users.split(",") if n.strip()]
    else:
        user_names = ["Owner"]

    # ── Load cascades ──
    cascades = load_cascades(log)
    log(f"Loaded {len(cascades)} cascade(s)")

    # ── Webcam ──
    try:
        cap = open_camera(args.camera, FRAME_WIDTH)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)

    log(f"Multi-user enrollment: {len(user_names)} user(s)")
    log(f"Samples per user: {args.samples}")

    # ── Enroll each user ──
    all_data: list = []
    for idx, name in enumerate(user_names, start=1):
        user_data = _enroll_user(log, cap, cascades, idx, name, args.samples)
        all_data.extend(user_data)

    cap.release()
    log(f"Total samples: {len(all_data)}")

    # ── Train ──
    log("Training LBPH model...")
    faces_only = [d[0] for d in all_data]
    labels_only = [d[1] for d in all_data]
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8,
    )
    recognizer.train(faces_only, np.array(labels_only))

    recognizer.write(args.output)
    log(f"Model saved to: {args.output}")

    # ── Auto-calibrate recognition threshold ──
    log("Calibrating recognition threshold...")
    confidences = []
    for sample in faces_only:
        _label, conf = recognizer.predict(sample)
        confidences.append(conf)
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))
    calibrated = max(min(mean_conf + 2.5 * std_conf, 95.0), 30.0)
    log(f"  Mean confidence: {mean_conf:.1f}")
    log(f"  Std deviation:   {std_conf:.1f}")
    log(f"  Calibrated threshold: {calibrated:.0f} (lower = stricter, default was 85)")

    # Save metadata
    try:
        meta_path = str(Path(args.output).with_suffix(".json"))
        meta = {
            "threshold": round(calibrated, 1),
            "mean_confidence": round(mean_conf, 1),
            "std_confidence": round(std_conf, 1),
            "samples": len(faces_only),
            "users": {str(idx): name for idx, name in enumerate(user_names, start=1)},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log(f"Metadata saved to: {meta_path}")
    except OSError as e:
        log(f"WARNING: Could not save metadata: {e}")

    log("")
    if len(user_names) > 1:
        log(f"Enrollment complete! {len(user_names)} users authorized: {', '.join(user_names)}")
    else:
        log("Enrollment complete! Run: python lock-on-absence.py")


if __name__ == "__main__":
    main()
