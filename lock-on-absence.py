#!/usr/bin/env python3
"""
Lock on Absence — Auto-lock your screen when you walk away.

Features:
  - Multi-angle face detection (frontal + left/right profile)
  - Optional facial recognition (only YOU prevent the lock)
  - Body presence detection (stays unlocked when you turn away)
  - Intruder detection (instant lock if someone else sits down)
  - Keep-awake mode (prevents sleep/lock while you're present)
  - Auto-calibrating body detection threshold

Setup:
  1. pip install -r requirements.txt
  2. (optional) python enroll.py            # train facial recognition
  3. python lock-on-absence.py              # start monitoring

Works on Windows, Linux, and macOS.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_utils import (
    BodyDetector,
    KeepAwake,
    Logger,
    StealthCamera,
    camera_available,
    detect_faces,
    install_signal_handlers,
    load_cascades,
    lock_screen,
    open_camera,
)

# ── Defaults (tunable via CLI) ─────────────────────────────────────
CHECK_INTERVAL = 1.5
ABSENCE_SECONDS = 10
CAMERA_INDEX = 0
SCALE_FACTOR = 1.03         # finer scan for distant/small faces
MIN_NEIGHBORS = 2           # more sensitive (streak confirmation prevents false locks)
MIN_FACE_SIZE = (30, 30)    # detects faces up to ~1.5m away
FRAME_WIDTH = 640
POST_LOCK_COOLDOWN = 30
RECOGNITION_THRESHOLD = 85  # more tolerant (lower = stricter)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(SCRIPT_DIR / "face_model.yml")


# ── Recognition ────────────────────────────────────────────────────

def recognize_owner(
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


# ── Main ───────────────────────────────────────────────────────────

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
    parser.add_argument(
        "--cooldown", type=int, default=POST_LOCK_COOLDOWN,
        help=f"Seconds of silence after locking (default: {POST_LOCK_COOLDOWN})",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Also write log output to a file",
    )
    parser.add_argument(
        "--max-body-only", type=int, default=60,
        help="Max seconds body detection keeps screen unlocked without face re-verification (default: 60)",
    )
    parser.add_argument(
        "--stealth", action="store_true",
        help="Open/close camera per frame to minimize LED glow (small CPU cost)",
    )
    parser.add_argument(
        "--meeting-pause", type=int, default=30,
        help="Seconds to pause monitoring when camera is in use by another app (default: 30)",
    )
    args = parser.parse_args()

    # ── Logger ──
    log = Logger(args.log_file)

    # ── Load cascades (frontal + profile) ──
    cascades = load_cascades(log)
    log(f"Loaded {len(cascades)} Haar cascade(s)")

    # ── Recognition model (optional) ──
    recognizer = None  # type: cv2.face.LBPHFaceRecognizer | None
    body_detector = BodyDetector()
    if os.path.exists(args.model):
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(args.model)
        recognizer = rec
        log(f"Face model loaded: {args.model}")
        log("Mode: OWNER RECOGNITION (only your face prevents lock)")
    else:
        log(f"No face model at {args.model} — run 'python enroll.py' first.")
        log("Mode: ANY FACE (any face prevents lock)")

    # ── Webcam ──
    if args.stealth:
        cap = StealthCamera(args.camera, FRAME_WIDTH)
        log("Camera mode: STEALTH (open/close per frame — LED blinks instead of solid)")
        # StealthCamera has no permanent VideoCapture; read() opens/closes per cycle
        _cap_obj = None  # used for release in finally
    else:
        try:
            _cap_obj = open_camera(args.camera, FRAME_WIDTH)
        except RuntimeError as exc:
            log(f"ERROR: {exc}")
            sys.exit(1)
        cap = _cap_obj

    # Camera busy tracking
    camera_busy_until: float = 0.0

    # ── Keep-awake ──
    keep_awake_mgr = KeepAwake(log)
    if args.no_keep_awake:
        log("Keep-awake: DISABLED")
        keep_awake_mgr = None  # type: ignore[assignment]

    # ── Graceful shutdown ──
    install_signal_handlers(cap, keep_awake_mgr or KeepAwake(), log)

    # ── Status ──
    lock_label = "DRY-RUN" if args.no_lock else "ACTIVE LOCK"
    awake_label = "keep-awake ON" if keep_awake_mgr else "keep-awake OFF"
    body_label = f"body-detect ON ({body_detector.status})" if recognizer else "body-detect OFF"
    max_body_label = f"max-body-only={args.max_body_only}s" if recognizer else ""
    stealth_label = " [STEALTH]" if args.stealth else ""
    log(f"Monitoring — delay={args.delay}s  interval={args.check_interval}s  cooldown={args.cooldown}s  {max_body_label}  [{lock_label}]  [{awake_label}]  [{body_label}]{stealth_label}")
    if args.stealth:
        log("LED: camera opens/closes per frame (~200ms blink instead of solid)")
    log("Press Ctrl+C to stop")

    # ── State machine ──
    absence_start: float | None = None
    locked_until: float = 0.0
    was_awake = False
    intruder_streak = 0
    last_face_time: float = time.time()  # assume owner starts present
    max_body_only: float = float(args.max_body_only) if recognizer else float("inf")

    # Track body-detect status for one-time log messages
    _body_detect_active = False

    try:
        while True:
            now = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                # Camera might be in use by another app (Teams, Zoom, etc.)
                if not args.stealth and not camera_available(args.camera):
                    if camera_busy_until == 0.0:
                        log(f"Camera in use by another app — pausing {args.meeting_pause}s (meeting mode)")
                    camera_busy_until = now + args.meeting_pause
                if camera_busy_until and now < camera_busy_until:
                    # Still waiting — retry camera
                    if not args.stealth:
                        try:
                            _cap_obj = open_camera(args.camera, FRAME_WIDTH)
                            cap = _cap_obj
                            camera_busy_until = 0.0
                            log("Camera available again — resuming monitoring")
                            continue
                        except RuntimeError:
                            pass
                time.sleep(2)
                continue
            else:
                if camera_busy_until:
                    camera_busy_until = 0.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(cascades, frame, SCALE_FACTOR, MIN_NEIGHBORS, MIN_FACE_SIZE)
            owner_present = False

            # ── Face detection ──
            if len(faces) == 0:
                owner_present = False
            elif recognizer is not None:
                for rect in faces:
                    is_owner, _conf = recognize_owner(recognizer, gray, rect)
                    if is_owner:
                        owner_present = True
                        break
            else:
                owner_present = True  # any face = owner

            # ── Cooldown after lock ──
            if locked_until and now < locked_until:
                time.sleep(args.check_interval)
                continue

            # ── Owner present ──
            if owner_present:
                intruder_streak = 0
                if _body_detect_active:
                    log("Owner face detected — body-detect disengaged")
                    _body_detect_active = False

                last_face_time = now  # reset re-verification timer

                if keep_awake_mgr and not was_awake:
                    keep_awake_mgr.enable()
                    was_awake = True

                # Update body reference frame periodically (only when face IS recognized)
                if recognizer is not None:
                    body_detector.update_ref(gray)

                if absence_start is not None:
                    log("Owner detected — timer reset")
                absence_start = None
                locked_until = 0.0

            # ── Owner NOT present ──
            else:
                if was_awake:
                    if keep_awake_mgr:
                        keep_awake_mgr.disable()
                    was_awake = False

                # Face detected but NOT owner → intruder check
                if len(faces) > 0 and recognizer is not None:
                    intruder_streak += 1
                    if intruder_streak >= 2:
                        if args.no_lock:
                            log(">>> [DRY-RUN] Would lock NOW (intruder confirmed)")
                        else:
                            log(">>> LOCKING NOW (intruder confirmed)")
                            lock_screen(keep_awake_mgr)
                        locked_until = now + args.cooldown
                        intruder_streak = 0
                        absence_start = None
                        _body_detect_active = False
                        log(f"Cooldown {args.cooldown}s...")
                else:
                    # No face at all — check body presence
                    intruder_streak = 0

                    if recognizer is not None and body_detector.present(gray):
                        # Body still in chair — but require periodic face re-verification
                        body_only_duration = now - last_face_time

                        if body_only_duration > max_body_only:
                            # Too long without face — lock for security
                            if args.no_lock:
                                log(f">>> [DRY-RUN] Would lock NOW (body-only timeout: {body_only_duration:.0f}s > {max_body_only:.0f}s)")
                            else:
                                log(f">>> LOCKING (body-only timeout: {body_only_duration:.0f}s > {max_body_only:.0f}s)")
                                lock_screen(keep_awake_mgr)
                            locked_until = now + args.cooldown
                            absence_start = None
                            _body_detect_active = False
                            log(f"Cooldown {args.cooldown}s...")
                        else:
                            if not _body_detect_active:
                                log(f"No face — body present, re-verify in {max_body_only - body_only_duration:.0f}s")
                                _body_detect_active = True
                            if absence_start is not None:
                                absence_start = None
                            if keep_awake_mgr and not was_awake:
                                keep_awake_mgr.enable()
                                was_awake = True
                            # NOTE: body reference is NOT updated here — only when face IS recognized
                            # This prevents an attacker's body from becoming the new reference
                    else:
                        if _body_detect_active:
                            log("Body no longer detected")
                            _body_detect_active = False

                        if absence_start is None:
                            absence_start = now
                            log(f"No face — waiting {args.delay}s...")
                        elif now - absence_start >= args.delay:
                            if args.no_lock:
                                log(">>> [DRY-RUN] Would lock now (nobody present)")
                            else:
                                log(">>> LOCKING (nobody present)")
                                lock_screen(keep_awake_mgr)
                            locked_until = now + args.cooldown
                            absence_start = None
                            log(f"Cooldown {args.cooldown}s...")

            time.sleep(args.check_interval)

    except KeyboardInterrupt:
        log("Stopped by user")
    finally:
        if keep_awake_mgr:
            keep_awake_mgr.disable()
        if not args.stealth and _cap_obj is not None:
            _cap_obj.release()
        elif args.stealth:
            cap.release()  # StealthCamera.release() is a no-op
        log("Cleanup complete")


if __name__ == "__main__":
    main()
