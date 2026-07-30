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
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from face_utils import (
    BodyDetector,
    EventLogger,
    KeepAwake,
    Logger,
    StealthCamera,
    YUNetDetector,
    camera_available,
    create_detector,
    detect_faces,
    detect_faces_dnn,
    download_yunet,
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
RECOGNITION_THRESHOLD = 65  # documented default LBPH confidence threshold

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(SCRIPT_DIR / "face_model.yml")


def recognize_owner(
    recognizer: cv2.face.LBPHFaceRecognizer,
    gray_frame: np.ndarray,
    face_rect: tuple,
    threshold: float = RECOGNITION_THRESHOLD,
) -> tuple[bool, float]:
    """
    Return (is_owner, confidence).
    confidence is LBPH distance — lower = better match.
    """
    x, y, w, h = face_rect
    roi = cv2.resize(gray_frame[y : y + h, x : x + w], (200, 200))
    _label, confidence = recognizer.predict(roi)
    result = confidence < threshold
    return result, confidence


def _do_lock(keep_awake_mgr, event_log, log, reason: str) -> None:
    """Lock screen with return-code verification and SIEM logging."""
    ok = lock_screen(keep_awake_mgr)
    if ok:
        return
    log(f"ERROR: lock_screen() returned False — screen may NOT be locked! ({reason})")
    event_log.lock_failed(reason)

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
    parser.add_argument(
        "--anti-spoof-timeout", type=int, default=0,
        help="Max seconds face can be perfectly still before locking (default: 0=off, suggested: 15)",
    )
    parser.add_argument(
        "--event-log", action="store_true",
        help="Write security events to Windows Event Log / Linux syslog",
    )
    parser.add_argument(
        "--siem", type=str, default=None,
        help="Write structured JSON events to this file for SIEM ingestion (Splunk, Sentinel, etc.)",
    )
    parser.add_argument(
        "--yunet", action="store_true",
        help="Use YuNet DNN face detector (downloads model if needed, much better than Haar)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Log face detection details (face count, confidence) for troubleshooting",
    )
    parser.add_argument(
        "--on-camera-failure", type=str, default="lock", choices=["lock", "warn"],
        help="Action when camera is unavailable: lock (fail-closed) or warn only (default: lock)",
    )
    parser.add_argument(
        "--any-face", action="store_true",
        help="Accept ANY detected face as owner (INSECURE — disables intruder detection)",
    )
    args = parser.parse_args()

    # ── Logger + Event log ──
    log = Logger(args.log_file)
    event_log = EventLogger(args.event_log, args.siem)
    if args.event_log:
        log(f"Event log: {'Windows Event Log' if sys.platform == 'win32' else 'syslog'} enabled")
    if args.siem:
        log(f"SIEM export: {args.siem} (JSON lines)")

    # ── Detector (YuNet or Haar) ──
    if args.yunet:
        try:
            yunet_path = download_yunet(str(SCRIPT_DIR))
            detector, detector_type = YUNetDetector(yunet_path), "yunet"
            log(f"Detector: YuNet DNN ({yunet_path})")
        except Exception as e:
            log(f"WARNING: YuNet download failed ({e}), falling back to Haar")
            detector, detector_type = load_cascades(log), "haar"
    else:
        detector, detector_type = load_cascades(log), "haar"
    if detector_type == "haar":
        log(f"Loaded {len(detector)} Haar cascade(s)")
    cascades = detector  # legacy alias for Haar path

    # ── Recognition model (optional) ──
    recognizer = None  # type: cv2.face.LBPHFaceRecognizer | None
    body_detector = BodyDetector()
    recognition_threshold = RECOGNITION_THRESHOLD
    if os.path.exists(args.model):
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(args.model)
        recognizer = rec
        log(f"Face model loaded: {args.model}")
        # Try to load calibrated threshold from face_model.json
        meta_path = str(Path(args.model).with_suffix(".json"))
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            recognition_threshold = meta.get("threshold", RECOGNITION_THRESHOLD)
            users_map = meta.get("users", {})
            if users_map:
                log(f"Authorized users: {', '.join(users_map.values())}")
            log(f"Recognition threshold: {recognition_threshold:.0f} (from face_model.json)")
        else:
            log(f"Recognition threshold: {recognition_threshold} (default — run enroll.py to create model)")
            log("WARNING: Run 'python enroll.py --samples 50' to calibrate for your face!")
        log("Mode: OWNER RECOGNITION (only your face prevents lock)")
    else:
        if not getattr(args, 'any_face', False):
            log("ERROR: No face model found and --any-face not set.")
            log("Run 'python enroll.py --samples 50' to create your face model.")
            log("Or use --any-face to accept ANY face (INSECURE — no intruder detection).")
            sys.exit(2)
        log("WARNING: --any-face mode — ANY face prevents lock (no intruder detection!)")
        log("Mode: ANY FACE (insecure — run enroll.py to enable owner recognition)")

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
    yunet_label = " [YuNet]" if detector_type == "yunet" else ""
    event_label = " [EventLog]" if args.event_log else ""
    log(f"Monitoring — delay={args.delay}s  interval={args.check_interval}s  cooldown={args.cooldown}s  {max_body_label}  [{lock_label}]  [{awake_label}]  [{body_label}]{stealth_label}{yunet_label}{event_label}")
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

    # Anti-spoof: track face movement (photo/video attacks)
    anti_spoof_timeout = float(args.anti_spoof_timeout) if recognizer else float("inf")
    prev_face_center: tuple[float, float] | None = None
    static_since: float | None = None
    if anti_spoof_timeout > 0 and anti_spoof_timeout != float("inf"):
        log(f"Anti-spoof: ON (max {anti_spoof_timeout:.0f}s static face -> lock)")
    elif recognizer:
        log("Anti-spoof: OFF (--anti-spoof-timeout 0)")

    try:
        _consecutive_fails = 0
        _first_fail_time: float | None = None
        _camera_fail_grace = 20.0  # seconds of tolerance before fail-closed
        _startup_grace = time.time() + 5  # 5s grace period before intruder checks
        while True:
            now = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                _consecutive_fails += 1
                if _first_fail_time is None:
                    _first_fail_time = now

                # ALWAYS release keep-awake — no proof of presence, no suppression
                if was_awake and keep_awake_mgr:
                    keep_awake_mgr.disable()
                    was_awake = False

                # Try to reopen camera (outside any pause window)
                if not args.stealth and _consecutive_fails % 3 == 0:
                    try:
                        _cap_obj = open_camera(args.camera, FRAME_WIDTH)
                        cap = _cap_obj
                        _consecutive_fails = 0
                        _first_fail_time = None
                        log("Camera recovered")
                        continue
                    except RuntimeError:
                        pass

                # Fail-closed watchdog: no signal for too long → lock
                if (args.on_camera_failure == "lock"
                        and _first_fail_time is not None
                        and now - _first_fail_time > _camera_fail_grace):
                    log(f">>> LOCKING (no camera signal for {now - _first_fail_time:.0f}s — fail-closed)")
                    _do_lock(keep_awake_mgr, event_log, log, "camera-failure")
                    event_log.camera_error(f"fail-closed lock after {now - _first_fail_time:.0f}s")
                    locked_until = now + args.cooldown
                    _first_fail_time = None
                    log(f"Cooldown {args.cooldown}s...")

                if _consecutive_fails == 1 or _consecutive_fails % 10 == 0:
                    log(f"Camera read failed ({_consecutive_fails}x) — retrying...")
                time.sleep(2)
                continue
            _consecutive_fails = 0
            _first_fail_time = None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces with current detector (YuNet or Haar)
            if detector_type == "yunet":
                faces = detect_faces_dnn(detector, frame)
            else:
                faces = detect_faces(detector, frame, SCALE_FACTOR, MIN_NEIGHBORS, MIN_FACE_SIZE)
            owner_present = False

            # ── Face detection ──
            if len(faces) == 0:
                owner_present = False
                owner_rect = None
            elif recognizer is not None:
                owner_rect = None
                for rect in faces:
                    is_owner, _conf = recognize_owner(recognizer, gray, rect, recognition_threshold)
                    if is_owner:
                        owner_present = True
                        owner_rect = rect
                        break
            else:
                owner_present = True  # any face = owner
                owner_rect = faces[0] if faces else None

            # Debug: log face count at most every 30s (tracks last emit time)
            _last_db = getattr(main, "_last_debug", 0.0)
            if args.debug and now - _last_db >= 28:  # ~30s with tolerance
                extra = ""
                if faces and recognizer:
                    _x, _y, _w, _h = faces[0]
                    _roi = cv2.resize(gray[_y:_y+_h, _x:_x+_w], (200, 200))
                    _l, _c = recognizer.predict(_roi)
                    extra = f", conf={_c:.0f}, thresh={recognition_threshold}"
                log(f"DEBUG: {len(faces)} face(s), owner={owner_present}{extra}")
                main._last_debug = now  # type: ignore[attr-defined]

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
                absence_start = None
                locked_until = 0.0

                # Heartbeat: confirm alive at most every 60s (tracks last emit time)
                _last_hb = getattr(main, "_last_heartbeat", 0.0)
                if now - _last_hb >= 58:  # ~60s with tolerance
                    log("Heartbeat — owner present, system active")
                    main._last_heartbeat = now  # type: ignore[attr-defined]
                    # Write watchdog heartbeat file
                    try:
                        with open(os.path.join(str(SCRIPT_DIR), "watchdog_heartbeat.txt"), "w") as _wfh:
                            _wfh.write(str(now))
                    except OSError:
                        pass

                # Anti-spoof: check face movement (photo/video has zero micro-movement)
                if owner_rect is not None and anti_spoof_timeout > 0 and anti_spoof_timeout != float("inf"):
                    x, y, w, h = owner_rect
                    cx, cy = x + w / 2.0, y + h / 2.0
                    # Use 1.5% of face width as threshold (~3px for 200px face, ~6px for 400px)
                    movement_threshold = max(w * 0.015, 4.0)
                    if prev_face_center is not None:
                        dx = abs(cx - prev_face_center[0])
                        dy = abs(cy - prev_face_center[1])
                        if dx < movement_threshold and dy < movement_threshold:  # essentially static
                            if static_since is None:
                                static_since = now
                            elif now - static_since > anti_spoof_timeout:
                                if args.no_lock:
                                    log(f">>> [DRY-RUN] Would lock NOW (anti-spoof: face static for {now - static_since:.0f}s)")
                                    event_log.spoof_lock(now - static_since)  # SIEM even in dry-run
                                else:
                                    log(f">>> LOCKING (anti-spoof: face static for {now - static_since:.0f}s — possible photo)")
                                    _do_lock(keep_awake_mgr, event_log, log, "anti-spoof")
                                    event_log.spoof_lock(now - static_since)
                                locked_until = now + args.cooldown
                                static_since = None
                                log(f"Cooldown {args.cooldown}s...")
                        else:
                            static_since = None  # moved — reset
                    prev_face_center = (cx, cy)

                if keep_awake_mgr and not was_awake:
                    keep_awake_mgr.enable()
                    was_awake = True

                # Update body reference frame periodically (only when face IS recognized)
                if recognizer is not None:
                    body_detector.update_ref(gray)
                    # Complete body detector calibration with OWNER present
                    if not body_detector.calibrated:
                        body_detector.complete_calibration()
                        log(f"Body detector calibrated — threshold={body_detector._threshold:.1f}")

                if absence_start is not None:
                    log("Owner detected — timer reset")

            # ── Owner NOT present ──
            else:
                # Reset anti-spoof tracking when no face is recognized
                prev_face_center = None
                static_since = None

                if was_awake:
                    if keep_awake_mgr:
                        keep_awake_mgr.disable()
                    was_awake = False

                # Face detected but NOT owner → intruder check (after grace period)
                if len(faces) > 0 and recognizer is not None:
                    if now < _startup_grace:
                        pass  # grace period — don't lock while user is settling in
                    else:
                        intruder_streak += 1
                        if intruder_streak >= 2:
                            if args.no_lock:
                                log(">>> [DRY-RUN] Would lock NOW (intruder confirmed)")
                                event_log.intruder_lock()  # always write SIEM even in dry-run
                            else:
                                log(">>> LOCKING NOW (intruder confirmed)")
                                _do_lock(keep_awake_mgr, event_log, log, "intruder")
                                event_log.intruder_lock()
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
                                event_log.body_timeout_lock(body_only_duration)
                            else:
                                log(f">>> LOCKING (body-only timeout: {body_only_duration:.0f}s > {max_body_only:.0f}s)")
                                _do_lock(keep_awake_mgr, event_log, log, "body-timeout")
                                event_log.body_timeout_lock(body_only_duration)
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
                                event_log.absence_lock(now - absence_start) if absence_start else None
                            else:
                                log(">>> LOCKING (nobody present)")
                                _do_lock(keep_awake_mgr, event_log, log, "absence")
                                event_log.absence_lock(now - absence_start) if absence_start else None
                            locked_until = now + args.cooldown
                            absence_start = None
                            log(f"Cooldown {args.cooldown}s...")

            time.sleep(args.check_interval)

    except KeyboardInterrupt:
        log("Stopped by user")
    except Exception as exc:
        log(f"CRASH: {exc}")
        event_log.camera_error(str(exc))
        if args.log_file:
            try:
                with open(args.log_file, "a") as f:
                    f.write(f"[CRASH] {datetime.now().strftime('%H:%M:%S')}\n")
                    traceback.print_exc(file=f)
            except OSError:
                pass
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
