#!/usr/bin/env python3
"""
Agent loop — a thin adapter around PresenceStateMachine.

Contract: this file contains NO presence logic. Its only jobs are
  1. read a frame,
  2. reduce it to an Observation,
  3. hand it to the state machine,
  4. execute the returned Verdict.

Every timer, every threshold and every one-shot log message lives in
state_machine.py. If you need to add an `if` about presence here, stop:
it belongs there, where the tests can reach it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .face_utils import (
    BodyDetector,
    EventLogger,
    KeepAwake,
    Logger,
    StealthCamera,
    YUNetDetector,
    camera_is_busy,
    detect_faces,
    download_yunet,
    file_sha256,
    load_cascades,
    lock_screen,
    open_camera,
    safe_face_roi,
)
from .state_machine import (
    Config,
    Mode,
    Observation,
    PresenceStateMachine,
    State,
    Verdict,
)

# ── Detection defaults ─────────────────────────────────────────────────
CHECK_INTERVAL = 1.5
FRAME_WIDTH = 640
SCALE_FACTOR = 1.05          # 1.03 was ~3.5x slower for no measured gain
MIN_NEIGHBORS = 3            # 2 produced frequent false faces (posters, texture)
MIN_FACE_SIZE = (30, 30)
RECOGNITION_THRESHOLD = 65   # see docs: calibrate with `loa-replay`, do not guess

HEARTBEAT_NAME = "watchdog_heartbeat.txt"


# ═══════════════════════════════════════════════════════════════════════
#  Vision helpers (pure functions — no state)
# ═══════════════════════════════════════════════════════════════════════

def recognize_faces(
    recognizer,
    gray: np.ndarray,
    faces: list,
    threshold: float,
) -> tuple[bool, tuple | None, float | None]:
    """
    Return (owner_found, owner_rect, best_confidence).

    ROIs are clamped: YuNet routinely returns negative coordinates for faces at
    the frame edge, and an unclamped slice produces an empty array that makes
    cv2.resize raise.
    """
    best_conf: float | None = None
    for rect in faces:
        roi = safe_face_roi(gray, rect)
        if roi is None:
            continue
        try:
            _label, conf = recognizer.predict(cv2.resize(roi, (200, 200)))
        except cv2.error:
            continue
        if best_conf is None or conf < best_conf:
            best_conf = conf
        if conf < threshold:
            return True, rect, conf
    return False, None, best_conf


def write_heartbeat(path: Path, log: Logger) -> None:
    """
    Atomically publish agent liveness.

    Written on EVERY tick, regardless of presence: the heartbeat proves the
    *agent* is alive, not that the *user* is there. Writing it only when the
    owner was present made the external watchdog fire during legitimate
    absences and race the user right after they logged back in.
    """
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(str(time.time()))
        os.replace(tmp, path)
    except OSError as exc:
        log(f"WARNING: could not write heartbeat: {exc}")


def notify_systemd(msg: str) -> None:
    """sd_notify without the systemd python bindings. No-op when not under systemd."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    import socket
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect("\0" + addr[1:] if addr.startswith("@") else addr)
            s.sendall(msg.encode())
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lock-on-absence",
        description="Lock the screen when you walk away. Webcam presence detection.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    g = p.add_argument_group("timing")
    g.add_argument("--delay", type=float, default=10.0,
                   help="Seconds with nobody present before locking (default: 10)")
    g.add_argument("--check-interval", type=float, default=CHECK_INTERVAL,
                   help=f"Seconds between samples (default: {CHECK_INTERVAL})")
    g.add_argument("--cooldown", type=float, default=30.0,
                   help="Quiet period after a lock (default: 30)")
    g.add_argument("--max-body-only", type=float, default=20.0,
                   help="Body-only may hold the screen this long (default: 20)")
    g.add_argument("--max-without-face", type=float, default=90.0,
                   help="Hard ceiling since last verified face (default: 90)")

    g = p.add_argument_group("policy")
    g.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.SECURITY.value,
                   help="security = fail-closed defaults; convenience = tolerant "
                        "(default: security)")
    g.add_argument("--on-camera-failure", choices=["lock", "warn"], default="lock",
                   help="Fail-closed (lock) or warn when the camera dies")
    g.add_argument("--camera-fail-grace", type=float, default=20.0,
                   help="Seconds of camera failure before fail-closed lock "
                        "(default: 20)")
    g.add_argument("--intruder-count", type=int, default=2,
                   help="Non-owner detections needed to lock (default: 2)")
    g.add_argument("--intruder-window", type=float, default=6.0,
                   help="Sliding window for those detections (default: 6)")
    g.add_argument("--startup-grace", type=float, default=5.0,
                   help="Suppress intruder locks for N seconds at start (default: 5)")
    g.add_argument("--meeting-pause", type=float, default=30.0,
                   help="Pause when another app holds the camera (default: 30)")
    g.add_argument("--anti-spoof-timeout", type=float, default=0.0,
                   help="Lock if the face is perfectly static this long. "
                        "0 = off. WEAK heuristic, not liveness detection.")
    g.add_argument("--any-face", action="store_true",
                   help="Accept ANY face as the owner. INSECURE: disables intruder "
                        "detection and body verification.")

    g = p.add_argument_group("hardware")
    g.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    g.add_argument("--model", type=str, default=None,
                   help="LBPH model path (default: ./face_model.yml)")
    g.add_argument("--yunet", action="store_true",
                   help="Use the YuNet DNN detector instead of Haar cascades")
    g.add_argument("--stealth", action="store_true",
                   help="Open/close the camera per frame. NOT RECOMMENDED: suppressing "
                        "the activity LED looks like spyware to EDR and to users.")
    g.add_argument("--no-keep-awake", action="store_true",
                   help="Never suppress the OS idle timeout")

    g = p.add_argument_group("output")
    g.add_argument("--no-lock", action="store_true",
                   help="Dry run: decide and log, never lock (events tagged dry_run)")
    g.add_argument("--log-file", type=str, default=None, help="Append log to this file")
    g.add_argument("--event-log", action="store_true",
                   help="Write to Windows Event Log / syslog")
    g.add_argument("--siem", type=str, default=None,
                   help="Append JSON-lines events to this file")
    g.add_argument("--debug", action="store_true", help="Log detection detail")
    return p


# ═══════════════════════════════════════════════════════════════════════
#  Setup
# ═══════════════════════════════════════════════════════════════════════

def _load_recognizer(args, log: Logger) -> tuple[object | None, float]:
    """Return (recognizer, threshold). Exits if no model and --any-face absent."""
    model = Path(args.model) if args.model else Path.cwd() / "face_model.yml"
    threshold = RECOGNITION_THRESHOLD

    if not model.exists():
        if not args.any_face:
            log(f"ERROR: no face model at {model}")
            log("Run 'lock-on-absence-enroll' first, or pass --any-face "
                "(INSECURE: any face keeps the screen unlocked).")
            sys.exit(2)
        log("WARNING: --any-face — ANY detected face keeps the screen unlocked. "
            "Intruder detection and body verification are DISABLED.")
        return None, threshold

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(str(model))
    log(f"Face model: {model}")

    meta_path = model.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log(f"WARNING: unreadable {meta_path.name} ({exc}) — using default threshold")
            return rec, threshold
        # Integrity check before trusting the model. A mismatch means the file
        # changed since enrollment: corruption, or someone swapping in their own
        # face. Either way, refusing to start beats authorising a stranger.
        expected = meta.get("model_sha256")
        if expected:
            try:
                actual = file_sha256(model)
            except OSError as exc:
                log(f"ERROR: cannot read model for integrity check: {exc}")
                sys.exit(3)
            if actual != expected:
                log(f"ERROR: {model.name} does not match the digest recorded at "
                    f"enrollment time.")
                log(f"  expected {expected[:16]}...  actual {actual[:16]}...")
                log("The model was modified or corrupted. Re-run "
                    "'lock-on-absence-enroll' to re-enroll, or restore a known-good "
                    "model. Refusing to start.")
                sys.exit(3)
            log("Model integrity: OK")
        else:
            log("NOTE: no model_sha256 in metadata (enrolled before v5.1) — "
                "integrity not verified. Re-run enrollment to enable the check.")

        raw = meta.get("threshold", threshold)
        # This file is user-writable. Without a range check anyone could set
        # threshold=1e9 and turn every face into the owner, silently.
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = threshold
        if not 20.0 <= raw <= 100.0:
            log(f"WARNING: threshold {raw} in {meta_path.name} is out of the sane "
                f"range [20,100] — ignoring it and using {threshold}")
        else:
            threshold = raw
        users = meta.get("users") or {}
        if users:
            log(f"Authorized: {', '.join(str(v) for v in users.values())}")
    log(f"Recognition threshold: {threshold:.0f}")
    return rec, threshold


def _build_config(args) -> Config:
    return Config(
        absence_delay=args.delay,
        max_body_only=args.max_body_only,
        max_without_face=args.max_without_face,
        intruder_count=args.intruder_count,
        intruder_window=args.intruder_window,
        cooldown=args.cooldown,
        meeting_pause=args.meeting_pause,
        startup_grace=args.startup_grace,
        anti_spoof_timeout=args.anti_spoof_timeout,
        on_camera_failure=args.on_camera_failure,
        camera_fail_grace=args.camera_fail_grace,
        mode=Mode(args.mode),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    log = Logger(args.log_file)
    event_log = EventLogger(args.event_log, args.siem, dry_run=args.no_lock)
    log(f"lock-on-absence {__version__}")

    recognizer, threshold = _load_recognizer(args, log)

    if args.yunet:
        try:
            detector, kind = YUNetDetector(download_yunet(str(Path.cwd()))), "yunet"
            log("Detector: YuNet DNN")
        except Exception as exc:
            log(f"WARNING: YuNet unavailable ({exc}) — falling back to Haar")
            detector, kind = load_cascades(log), "haar"
    else:
        detector, kind = load_cascades(log), "haar"

    if args.stealth:
        cap, cap_obj = StealthCamera(args.camera, FRAME_WIDTH), None
        log("Camera: STEALTH mode (per-frame open/close)")
    else:
        try:
            cap_obj = open_camera(args.camera, FRAME_WIDTH)
        except RuntimeError as exc:
            log(f"ERROR: {exc}")
            return 1
        cap = cap_obj

    keep_awake = None if args.no_keep_awake else KeepAwake(log)
    body = BodyDetector()
    cfg = _build_config(args)
    psm = PresenceStateMachine(cfg)
    st = State()
    heartbeat = Path.cwd() / HEARTBEAT_NAME

    log(f"Mode: {cfg.mode.value} | on-camera-failure={cfg.on_camera_failure} | "
        f"delay={cfg.absence_delay:.0f}s | body-only={cfg.max_body_only:.0f}s | "
        f"ceiling={cfg.max_without_face:.0f}s | "
        f"{'DRY-RUN' if args.no_lock else 'ACTIVE LOCK'}")
    for clamp in cfg.clamps:
        log(f"NOTE: {clamp}")
    if cfg.anti_spoof_timeout > 0:
        log("NOTE: --anti-spoof-timeout is a weak heuristic, not liveness detection.")
    log("Ctrl+C to stop")

    awake = False
    last_debug = 0.0
    notify_systemd("READY=1")

    try:
        while True:
            now = time.monotonic()

            # ── read ──────────────────────────────────────────────────
            try:
                ok, frame = cap.read()
            except Exception as exc:                      # a driver hiccup is not fatal
                log(f"WARNING: camera read raised {type(exc).__name__}: {exc}")
                ok, frame = False, None
            ok = bool(ok) and frame is not None

            # ── reduce to an Observation ───────────────────────────────
            faces: list = []
            owner, rect, conf = False, None, None
            gray = None
            if ok:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = (detector.detect(frame) if kind == "yunet"
                             else detect_faces(detector, frame, SCALE_FACTOR,
                                               MIN_NEIGHBORS, MIN_FACE_SIZE))
                    if recognizer is not None and faces:
                        owner, rect, conf = recognize_faces(
                            recognizer, gray, faces, threshold)
                except cv2.error as exc:
                    log(f"WARNING: frame analysis failed: {exc}")
                    ok, faces = False, []

            obs = Observation(
                t=now,
                faces=len(faces),
                owner_recognized=owner,
                scene_unchanged=bool(ok and recognizer is not None
                                     and gray is not None and body.present(gray)),
                camera_ok=ok,
                camera_busy=(not ok) and camera_is_busy(args.camera),
                has_recognizer=recognizer is not None,
                face_center=((rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)
                             if rect is not None else None),
                face_width=float(rect[2]) if rect is not None else 0.0,
            )

            # ── decide ────────────────────────────────────────────────
            verdict: Verdict = psm.step(obs, st)

            # ── execute ───────────────────────────────────────────────
            if verdict.message:
                log(verdict.message)

            if verdict.is_lock:
                if args.no_lock:
                    log(f"[DRY-RUN] would lock: {verdict.reason.value}")
                    locked = True
                else:
                    locked = lock_screen(keep_awake)
                    if not locked:
                        log("ERROR: every lock mechanism failed — "
                            "the screen is NOT locked")
                event_log.lock(verdict.reason.value, ok=locked, **verdict.detail)
                awake = False
            else:
                if keep_awake is not None and verdict.keep_awake != awake:
                    keep_awake.enable() if verdict.keep_awake else keep_awake.disable()
                    awake = verdict.keep_awake

            # Body reference and calibration only ever advance on a verified
            # face, so an intruder's body can never become the reference.
            if owner and gray is not None and recognizer is not None:
                body.update_ref(gray)
                body.sample_noise(gray)
                if not body.calibrated and body.complete_calibration():
                    log(f"Body detector calibrated — threshold={body.threshold:.1f}")

            write_heartbeat(heartbeat, log)
            notify_systemd("WATCHDOG=1")

            if args.debug and now - last_debug >= 28:
                last_debug = now
                log(f"DEBUG faces={len(faces)} owner={owner} "
                    f"conf={f'{conf:.0f}' if conf is not None else '-'} "
                    f"thresh={threshold:.0f} decision={verdict.decision.value} "
                    f"reason={verdict.reason.value} "
                    f"since_face={now - (st.last_proof_of_presence or now):.0f}s")

            # Recover the device when it comes back rather than looping on a
            # dead handle forever.
            if not ok and not args.stealth and st.camera_fail_streak in (3, 9, 27):
                try:
                    cap_obj = open_camera(args.camera, FRAME_WIDTH)
                    cap = cap_obj
                    psm.reset_camera_failure(st)
                    log("Camera recovered")
                except RuntimeError:
                    pass

            time.sleep(2.0 if not ok else args.check_interval)

    except KeyboardInterrupt:
        log("Stopped by user")
        return 0
    except Exception as exc:
        log(f"CRASH: {type(exc).__name__}: {exc}")
        event_log.agent_crash(f"{type(exc).__name__}: {exc}")
        if args.log_file:
            try:
                with open(args.log_file, "a", encoding="utf-8") as fh:
                    traceback.print_exc(file=fh)
            except OSError:
                pass
        # Fail closed: a dead agent must not leave the session unlocked.
        if not args.no_lock:
            log("Fail-closed: locking before exit")
            lock_screen(keep_awake)
        return 1
    finally:
        if keep_awake is not None:
            keep_awake.disable()
        if cap_obj is not None:
            cap_obj.release()
        log("Cleanup complete")


if __name__ == "__main__":
    sys.exit(main())
