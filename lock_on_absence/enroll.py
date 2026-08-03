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
    python enroll.py --purge                   # delete model + metadata

Tips:
    Move your head: front -> left profile -> right profile -> up -> down.
    Good lighting and a clean background improve accuracy.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .face_utils import (
    Logger,
    detect_faces,
    estimate_angle,
    file_sha256,
    load_cascades,
    open_camera,
    restrict_file_permissions,
)

# ── Defaults ───────────────────────────────────────────────────────
SAMPLES = 30
CAMERA_INDEX = 0
FRAME_WIDTH = 640
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = str(SCRIPT_DIR / "face_model.yml")
RECOGNITION_THRESHOLD = 65  # documented default LBPH confidence threshold


def _enroll_user(log: Logger, cap: cv2.VideoCapture, cascades: list,
                 user_label: int, user_name: str, samples_needed: int) -> list:
    """Capture face samples for one user. Returns list of (roi, label) tuples."""
    log("")
    log(f"=== User {user_label}: {user_name} ===")
    log("Move your head through all angles:")
    log("  -> FRONT -> LEFT profile -> RIGHT profile -> UP -> DOWN")
    log("")

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
            roi = _safe_roi(gray, (x, y, w, h))
            if roi is None:
                continue
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


def _safe_roi(gray: np.ndarray, rect: tuple) -> np.ndarray | None:
    """Extract face ROI with boundary clamping.
    YuNet can return negative coordinates at frame edges (P1-5)."""
    H, W = gray.shape[:2]
    x, y, w, h = rect
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return gray[y0:y1, x0:x1]


def _self_test(log: Logger) -> int:
    """
    Prove the SFace pipeline works on THIS machine before trusting it.

    Exists because the ONNX lives in git-lfs: a naive fetch returns a ~130-byte
    pointer, cv2 then fails with an opaque ONNX parse error, and since the file
    exists on disk every later run reuses the broken copy. 30 seconds here saves
    an afternoon of confusion.
    """
    import numpy as _np

    from .recognizers import (
        SFACE_COSINE_THRESHOLD,
        SFACE_MIN_BYTES,
        download_sface,
    )

    log("SFace self-test")
    try:
        path = download_sface(".", log=log)
    except RuntimeError as exc:
        log(f"FAIL: {exc}")
        return 1

    size = os.path.getsize(path)
    log(f"  [ok] model present: {size / 1_048_576:.1f} MB")
    if size < SFACE_MIN_BYTES:
        log("FAIL: file is too small to be the model (git-lfs pointer?)")
        return 1

    try:
        sf = cv2.FaceRecognizerSF.create(path, "")
    except cv2.error as exc:
        log(f"FAIL: cv2 could not load the ONNX: {exc}")
        log("      Delete the file and re-run; it is probably a truncated download.")
        return 1
    log("  [ok] cv2.FaceRecognizerSF.create")

    rng = _np.random.default_rng(0)
    img = rng.integers(0, 255, (112, 112, 3), dtype=_np.uint8)
    feat = sf.feature(img)
    dims = _np.asarray(feat).reshape(1, -1).shape[1]
    if dims != 128:
        log(f"FAIL: expected a 128-d embedding, got {dims}")
        return 1
    log(f"  [ok] embedding is {dims}-d")

    self_sim = sf.match(feat, feat, cv2.FaceRecognizerSF_FR_COSINE)
    if abs(self_sim - 1.0) > 1e-3:
        log(f"FAIL: self-similarity should be 1.0, got {self_sim:.4f}")
        return 1
    log(f"  [ok] self-similarity {self_sim:.5f}")

    other = sf.feature(rng.integers(0, 255, (112, 112, 3), dtype=_np.uint8))
    cross = sf.match(feat, other, cv2.FaceRecognizerSF_FR_COSINE)
    log(f"  [--] two random images score {cross:.4f} "
        f"(threshold {SFACE_COSINE_THRESHOLD}) — noise, not faces; "
        f"informational only")

    log("")
    log("SFace works on this machine. Now enroll:")
    log("    lock-on-absence-enroll --recognizer sface")
    log("Then MEASURE before switching the default — the published threshold of "
        f"{SFACE_COSINE_THRESHOLD} is a starting point, not a calibration for your "
        "camera:")
    log("    lock-on-absence-replay --video mesa.mp4 --labels mesa.csv")
    return 0


def main() -> int:
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
        help="Comma-separated user names (e.g. 'Alice,Bob'). Without: interactive enrollment of 1 user",
    )
    parser.add_argument(
        "--purge", action="store_true",
        help="Delete face model and metadata (revoke all enrolled users)",
    )
    parser.add_argument(
        "--no-consent", action="store_true",
        help="Skip consent prompt (for automated enrollment; use with caution)",
    )
    parser.add_argument(
        "--recognizer", choices=["lbph", "sface"], default="sface",
        help="Backend to enroll for. Default 'sface': embedding model with a "
             "published operating point (cosine 0.363, validated on LFW/CFP-FP/"
             "AgeDB) that transfers across lighting. 'lbph' is the legacy "
             "backend, kept for one release; its threshold does not transfer and "
             "was never measured.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Verify the SFace ONNX loads and embeds on THIS machine, then exit. "
             "Run this once after install: the model is git-lfs and a truncated "
             "download fails with an opaque parse error.",
    )
    args = parser.parse_args()

    log = Logger()

    # ── Handle --purge ──
    if args.self_test:
        return _self_test(log)

    if args.purge:
        model_path = Path(args.output)
        meta_path = model_path.with_suffix(".json")
        deleted_any = False
        for p in [model_path, meta_path]:
            if p.exists():
                p.unlink()
                log(f"Deleted: {p}")
                deleted_any = True
        if deleted_any:
            log("Face data purged — all enrolled users revoked.")
        else:
            log("No face data found — nothing to purge.")
        sys.exit(0)

    # ── Consent prompt ──
    if not args.no_consent:
        log("")
        log("By enrolling your face, you consent to:")
        log("  - Periodic webcam captures while monitoring is active")
        log("  - Storage of facial recognition data (LBPH model) on this machine")
        log("  - Logging of events (presence, absence, intruder) to local files and SIEM")
        log("")
        try:
            input("Press Enter to continue, or Ctrl+C to cancel...")
        except (EOFError, KeyboardInterrupt):
            log("")
            log("Enrollment cancelled.")
            sys.exit(0)

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
        if idx > 1:
            log(f"(5s pause — {name}, sit down now)")
            time.sleep(5)
        user_data = _enroll_user(log, cap, cascades, idx, name, args.samples)
        all_data.extend(user_data)

    cap.release()
    log(f"Total samples: {len(all_data)}")

    # ── Train ──
    log("Training LBPH model...")
    faces_only = [d[0] for d in all_data]
    labels_only = [d[1] for d in all_data]
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=8, grid_x=6, grid_y=6,
    )
    recognizer.train(faces_only, np.array(labels_only))

    recognizer.write(args.output)
    log(f"Model saved to: {args.output}")

    # Restrict model file permissions (P1-4). Reports honestly on failure:
    # os.chmod is close to a no-op on Windows, where ACLs are the real mechanism.
    ok, how = restrict_file_permissions(args.output)
    log(f"Model permissions: {how}" if ok
        else f"WARNING: could not restrict model permissions — {how}")

    # Integrity digest. This is tamper DETECTION, not prevention: an attacker who
    # can rewrite face_model.yml can usually rewrite face_model.json too. What it
    # buys is (a) catching corruption, and (b) forcing any tamper to be
    # consistent across two files instead of one. Real prevention needs the key
    # in an OS keyring — see MIGRATION notes.
    model_digest = file_sha256(args.output)

    # Save metadata (user names, sample count)
    try:
        meta_path = str(Path(args.output).with_suffix(".json"))
        meta = {
            "threshold": RECOGNITION_THRESHOLD,
            "model_sha256": model_digest,
            "samples": len(faces_only),
            "users": {str(idx): name for idx, name in enumerate(user_names, start=1)},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        restrict_file_permissions(meta_path)
        log(f"Metadata saved to: {meta_path}")
        log(f"Recognition threshold: {RECOGNITION_THRESHOLD} (default — edit face_model.json to tune)")
    except OSError as e:
        log(f"WARNING: Could not save metadata: {e}")

    log("")
    if len(user_names) > 1:
        log(f"Enrollment complete! {len(user_names)} users authorized: {', '.join(user_names)}")
    else:
        log("Enrollment complete! Run: python lock-on-absence.py")



    return 0
if __name__ == "__main__":
    sys.exit(main())
