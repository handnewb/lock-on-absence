#!/usr/bin/env python3
"""
FAR/FRR harness — measure the thing instead of guessing it.

Every threshold in this project was tuned by feel: the recognition threshold
went 85 -> 60 -> 30 -> 55 -> 65 in a single day, minNeighbors 3 -> 2, scaleFactor
1.05 -> 1.03, all without a single measured number. This module exists so that
stops. Nothing gets tuned again without a before/after table.

Three modes:

  1. scenario  — replay a .jsonl of Observations through the state machine.
                 Pure logic, no OpenCV, no camera, deterministic. Runs in CI.

  2. video     — replay a real recording through the full vision pipeline with
                 timestamps injected from the frame index, so a 30-minute clip
                 evaluates in seconds and always the same way.

  3. record    — run mode 2 but emit a scenario file instead of metrics, so you
                 can iterate on the state machine without re-running detection.

Definitions used here (stated explicitly, because everyone means something
slightly different by FAR/FRR):

  FRR  false reject rate — fraction of `owner` intervals in which the screen
       locked. A lock while the legitimate user sits there is the error that
       makes people uninstall the tool.

  FAR  false accept rate — fraction of `intruder` intervals in which the screen
       did NOT lock within --far-window seconds. This is the security failure.

  TTL  time-to-lock — seconds from the start of an `absent` interval to the
       lock. Reported as median and p90.

Label file (CSV, header optional):

    start_sec,end_sec,truth
    0,45,owner
    45,60,absent
    60,75,intruder
    75,120,body_only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__
from .state_machine import (
    Config,
    Decision,
    Mode,
    Observation,
    PresenceStateMachine,
    State,
)

TRUTHS = ("owner", "intruder", "absent", "body_only")


# ═══════════════════════════════════════════════════════════════════════
#  Ground truth
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Interval:
    start: float
    end: float
    truth: str

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def load_labels(path: Path) -> list[Interval]:
    rows: list[Interval] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.reader(fh):
            if not raw or raw[0].lstrip().startswith("#"):
                continue
            if raw[0].strip().lower() in ("start", "start_sec"):
                continue          # header
            if len(raw) < 3:
                raise ValueError(f"label row needs 3 columns, got {raw!r}")
            truth = raw[2].strip().lower()
            if truth not in TRUTHS:
                raise ValueError(f"unknown truth {truth!r}; expected one of {TRUTHS}")
            rows.append(Interval(float(raw[0]), float(raw[1]), truth))
    rows.sort(key=lambda i: i.start)
    for a, b in itertools.pairwise(rows):
        if b.start < a.end:
            raise ValueError(f"labels overlap: {a} and {b}")
    if not rows:
        raise ValueError(f"{path} contains no labelled intervals")
    return rows


def truth_at(intervals: list[Interval], t: float) -> str | None:
    for iv in intervals:
        if iv.contains(t):
            return iv.truth
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Results
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LockEvent:
    t: float
    reason: str
    truth: str | None


@dataclass
class Report:
    duration: float = 0.0
    ticks: int = 0
    locks: list[LockEvent] = field(default_factory=list)

    owner_intervals: int = 0
    owner_intervals_with_lock: int = 0
    owner_seconds: float = 0.0

    intruder_intervals: int = 0
    intruder_intervals_missed: int = 0

    absent_intervals: int = 0
    time_to_lock: list[float] = field(default_factory=list)
    absent_intervals_missed: int = 0

    body_intervals: int = 0
    body_intervals_with_lock: int = 0

    # ── derived ────────────────────────────────────────────────────────
    @property
    def frr(self) -> float | None:
        if not self.owner_intervals:
            return None
        return self.owner_intervals_with_lock / self.owner_intervals

    @property
    def far(self) -> float | None:
        if not self.intruder_intervals:
            return None
        return self.intruder_intervals_missed / self.intruder_intervals

    @property
    def spurious_locks_per_hour(self) -> float | None:
        if self.owner_seconds <= 0:
            return None
        n = sum(1 for e in self.locks if e.truth == "owner")
        return n * 3600.0 / self.owner_seconds

    @property
    def ttl_median(self) -> float | None:
        return statistics.median(self.time_to_lock) if self.time_to_lock else None

    @property
    def ttl_p90(self) -> float | None:
        if not self.time_to_lock:
            return None
        s = sorted(self.time_to_lock)
        return s[min(len(s) - 1, round(0.9 * (len(s) - 1)))]

    def lock_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.locks:
            out[e.reason] = out.get(e.reason, 0) + 1
        return out

    # ── rendering ──────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "duration_seconds": round(self.duration, 1),
            "ticks": self.ticks,
            "far": None if self.far is None else round(self.far, 4),
            "frr": None if self.frr is None else round(self.frr, 4),
            "spurious_locks_per_hour": (
                None if self.spurious_locks_per_hour is None
                else round(self.spurious_locks_per_hour, 2)),
            "time_to_lock_median": (
                None if self.ttl_median is None else round(self.ttl_median, 1)),
            "time_to_lock_p90": (
                None if self.ttl_p90 is None else round(self.ttl_p90, 1)),
            "absence_misses": self.absent_intervals_missed,
            "body_intervals_with_lock": self.body_intervals_with_lock,
            "locks_by_reason": self.lock_counts(),
            "intervals": {
                "owner": self.owner_intervals,
                "intruder": self.intruder_intervals,
                "absent": self.absent_intervals,
                "body_only": self.body_intervals,
            },
        }

    def render(self) -> str:
        def pct(v: float | None) -> str:
            return "  n/a" if v is None else f"{v * 100:5.1f}%"

        def sec(v: float | None) -> str:
            return "  n/a" if v is None else f"{v:5.1f}s"

        lines = [
            "",
            "=== FAR / FRR report ===",
            f"  replay duration      {self.duration:8.1f}s over {self.ticks} ticks",
            "",
            f"  FAR  intruder missed {pct(self.far)}   "
            f"({self.intruder_intervals_missed}/{self.intruder_intervals} intervals)",
            f"  FRR  owner rejected  {pct(self.frr)}   "
            f"({self.owner_intervals_with_lock}/{self.owner_intervals} intervals)",
            "",
            f"  spurious locks/hour  {'  n/a' if self.spurious_locks_per_hour is None else f'{self.spurious_locks_per_hour:5.2f}'}",
            f"  time-to-lock median  {sec(self.ttl_median)}",
            f"  time-to-lock p90     {sec(self.ttl_p90)}",
            f"  absence never locked {self.absent_intervals_missed}/"
            f"{self.absent_intervals} intervals",
            f"  body-only locked in  {self.body_intervals_with_lock}/"
            f"{self.body_intervals} intervals",
            "",
            "  locks by reason:",
        ]
        counts = self.lock_counts()
        if counts:
            for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {reason:<16} {n}")
        else:
            lines.append("    (none)")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Core replay
# ═══════════════════════════════════════════════════════════════════════

def replay(
    observations: Iterable[Observation],
    labels: list[Interval],
    cfg: Config,
    far_window: float = 5.0,
) -> Report:
    """
    Push observations through the state machine and score against ground truth.

    The state machine only ever sees obs.t, so a video replayed at any speed
    produces byte-identical results. That determinism is the whole point.
    """
    psm = PresenceStateMachine(cfg)
    st = State()
    rep = Report()

    locked_in_interval: dict[int, list[float]] = {}
    last_t = 0.0

    for obs in observations:
        rep.ticks += 1
        last_t = obs.t
        verdict = psm.step(obs, st)
        if verdict.decision is not Decision.LOCK:
            continue
        truth = truth_at(labels, obs.t)
        rep.locks.append(LockEvent(obs.t, verdict.reason.value, truth))
        for idx, iv in enumerate(labels):
            if iv.contains(obs.t):
                locked_in_interval.setdefault(idx, []).append(obs.t)
                break

    rep.duration = last_t

    for idx, iv in enumerate(labels):
        hits = locked_in_interval.get(idx, [])
        if iv.truth == "owner":
            rep.owner_intervals += 1
            rep.owner_seconds += iv.duration
            if hits:
                rep.owner_intervals_with_lock += 1
        elif iv.truth == "intruder":
            rep.intruder_intervals += 1
            # A lock counts only if it lands inside the tolerance window.
            if not any(h - iv.start <= far_window for h in hits):
                rep.intruder_intervals_missed += 1
        elif iv.truth == "absent":
            rep.absent_intervals += 1
            if hits:
                rep.time_to_lock.append(hits[0] - iv.start)
            else:
                rep.absent_intervals_missed += 1
        elif iv.truth == "body_only":
            rep.body_intervals += 1
            if hits:
                rep.body_intervals_with_lock += 1

    return rep


# ═══════════════════════════════════════════════════════════════════════
#  Mode 1 — scenario files
# ═══════════════════════════════════════════════════════════════════════

def read_scenario(path: Path) -> Iterator[Observation]:
    """One JSON object per line; unknown keys rejected loudly, not silently."""
    allowed = set(Observation.__dataclass_fields__)
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: bad JSON: {exc}") from exc
            extra = set(raw) - allowed
            if extra:
                raise ValueError(f"{path}:{lineno}: unknown field(s) {sorted(extra)}")
            if "face_center" in raw and raw["face_center"] is not None:
                raw["face_center"] = tuple(raw["face_center"])
            yield Observation(**raw)


def write_scenario(path: Path, observations: Iterable[Observation]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for obs in observations:
            d = asdict(obs)
            if d.get("face_center") is not None:
                d["face_center"] = list(d["face_center"])
            fh.write(json.dumps(d) + "\n")
            n += 1
    return n


# ═══════════════════════════════════════════════════════════════════════
#  Synthetic scenarios — so CI has something to measure without a webcam
# ═══════════════════════════════════════════════════════════════════════

def synth_scenario(
    interval: float = 1.5,
    detect_flicker: float = 0.0,
    recognizer: bool = True,
    repeat: int = 3,
) -> tuple[list[Observation], list[Interval]]:
    """
    Build a canonical 300-second scenario with matching ground truth.

    `detect_flicker` drops that fraction of detections (deterministically) to
    model Haar losing the face at an angle. It is the knob that exposed the
    consecutive-streak evasion: with flicker, a streak counter never reaches 2
    and the intruder is never locked out.
    """
    # Ordering matters: an `intruder` segment is only meaningful when the screen
    # is unlocked at that moment, so each one follows a long `owner` segment with
    # enough headroom for the previous lock's cooldown to have expired. Getting
    # this wrong is how the first draft measured FAR=100% against itself.
    #
    # The block is repeated so FAR/FRR have resolution: with a single intruder
    # interval the rate can only ever read 0% or 100%, which measures nothing.
    block: list[tuple[float, str]] = [
        (60.0, "owner"),       # settle in, cooldown from the previous cycle expires
        (30.0, "absent"),      # -> absence lock
        (80.0, "owner"),       # recover
        (12.0, "intruder"),    # short: someone sits down briefly. Must be caught.
        (70.0, "owner"),       # recover
        (50.0, "body_only"),   # -> body-only timeout
        (60.0, "owner"),       # recover
    ]
    labels: list[Interval] = []
    cursor = 0.0
    for _ in range(max(1, repeat)):
        for dur, truth in block:
            labels.append(Interval(cursor, cursor + dur, truth))
            cursor += dur

    obs: list[Observation] = []
    tick = 0
    t = 0.0
    end = cursor
    while t < end:
        truth = truth_at(labels, t) or "absent"
        tick += 1
        # Deterministic pseudo-flicker: no RNG, so replays stay reproducible.
        drop = detect_flicker > 0 and ((tick * 7919) % 1000) / 1000.0 < detect_flicker

        if truth == "owner":
            o = Observation(t=t, faces=0 if drop else 1,
                            owner_recognized=not drop,
                            scene_unchanged=True, camera_ok=True,
                            has_recognizer=recognizer,
                            face_center=(320.0 + (tick % 5), 240.0 + (tick % 3)),
                            face_width=180.0)
        elif truth == "intruder":
            o = Observation(t=t, faces=0 if drop else 1, owner_recognized=False,
                            scene_unchanged=False, camera_ok=True,
                            has_recognizer=recognizer,
                            face_center=(300.0, 250.0), face_width=170.0)
        elif truth == "body_only":
            o = Observation(t=t, faces=0, owner_recognized=False,
                            scene_unchanged=True, camera_ok=True,
                            has_recognizer=recognizer)
        else:  # absent
            o = Observation(t=t, faces=0, owner_recognized=False,
                            scene_unchanged=False, camera_ok=True,
                            has_recognizer=recognizer)
        obs.append(o)
        t += interval
    return obs, labels


# ═══════════════════════════════════════════════════════════════════════
#  Mode 2/3 — video
# ═══════════════════════════════════════════════════════════════════════

def observations_from_video(
    video: Path,
    interval: float,
    model: Path | None,
    threshold: float,
    use_yunet: bool,
    log=print,
) -> Iterator[Observation]:
    """
    Decode a recording and emit one Observation per sampling interval.

    Timestamps come from frame_index / fps — never from wall clock — so the
    result is identical on a fast laptop and a loaded CI runner.
    """
    import cv2  # imported here so scenario mode never needs OpenCV

    from .face_utils import BodyDetector, detect_faces, load_cascades, safe_face_roi

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * interval))
    log(f"video={video.name} fps={fps:.1f} sampling every {step} frames "
        f"(~{step / fps:.2f}s)")

    recognizer = None
    if model and model.exists():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(model))
        log(f"recognizer={model.name} threshold={threshold:.0f}")
    else:
        log("no model — every detected face counts as owner (--any-face equivalent)")

    detector = load_cascades()
    if use_yunet:
        try:
            from .face_utils import YUNetDetector, download_yunet
            detector = YUNetDetector(download_yunet("."))
            log("detector=YuNet DNN")
        except Exception as exc:
            log(f"YuNet unavailable ({exc}) — using Haar cascades")

    body = BodyDetector()
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            idx += 1
            if not ok or frame is None:
                continue
            t = (idx - 1) / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = (detector.detect(frame) if hasattr(detector, "detect")
                     else detect_faces(detector, frame))
            owner, rect = False, None
            if recognizer is not None:
                for r in faces:
                    roi = safe_face_roi(gray, r)
                    if roi is None:
                        continue
                    try:
                        _l, conf = recognizer.predict(cv2.resize(roi, (200, 200)))
                    except cv2.error:
                        continue
                    if conf < threshold:
                        owner, rect = True, r
                        break
            elif faces:
                owner, rect = True, faces[0]

            scene = bool(recognizer is not None and body.present(gray))
            if owner:
                body.update_ref(gray)
                body.sample_noise(gray)
                body.complete_calibration()

            yield Observation(
                t=t, faces=len(faces), owner_recognized=owner,
                scene_unchanged=scene, camera_ok=True,
                has_recognizer=recognizer is not None,
                face_center=((rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)
                             if rect is not None else None),
                face_width=float(rect[2]) if rect is not None else 0.0,
            )
    finally:
        cap.release()


# ═══════════════════════════════════════════════════════════════════════
#  Sweep
# ═══════════════════════════════════════════════════════════════════════

def sweep(observations: list[Observation], labels: list[Interval],
          base: Config, field_name: str, values: list[float],
          far_window: float) -> str:
    """Re-score the same observations under different config values."""
    rows = [f"  sweep {field_name}",
            f"  {'value':>10} {'FAR':>8} {'FRR':>8} {'TTL med':>9} {'spur/h':>8}"]
    for v in values:
        cfg = Config(**{**{k: getattr(base, k) for k in Config.__dataclass_fields__},
                        field_name: v})
        rep = replay(observations, labels, cfg, far_window)
        far = "n/a" if rep.far is None else f"{rep.far * 100:.1f}%"
        frr = "n/a" if rep.frr is None else f"{rep.frr * 100:.1f}%"
        ttl = "n/a" if rep.ttl_median is None else f"{rep.ttl_median:.1f}s"
        sph = ("n/a" if rep.spurious_locks_per_hour is None
               else f"{rep.spurious_locks_per_hour:.2f}")
        rows.append(f"  {v:>10} {far:>8} {frr:>8} {ttl:>9} {sph:>8}")
    return "\n".join(rows) + "\n"


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lock-on-absence-replay",
        description="Measure FAR/FRR by replaying video or scenarios "
                    "through the state machine.",
        epilog="Examples:\n"
               "  lock-on-absence-replay --synthetic\n"
               "  lock-on-absence-replay --synthetic --flicker 0.4 "
               "--sweep intruder_window=3,6,12\n"
               "  lock-on-absence-replay --scenario day.jsonl --labels day.csv\n"
               "  lock-on-absence-replay --video desk.mp4 --labels desk.csv "
               "--model face_model.yml\n"
               "  lock-on-absence-replay --video desk.mp4 --record desk.jsonl\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    src = p.add_argument_group("input (pick one)")
    src.add_argument("--scenario", type=Path, help="Replay a .jsonl of Observations")
    src.add_argument("--video", type=Path, help="Replay a recording")
    src.add_argument("--synthetic", action="store_true",
                     help="Use the built-in 300s scenario (no files needed)")

    p.add_argument("--labels", type=Path,
                   help="Ground-truth CSV: start_sec,end_sec,truth")
    p.add_argument("--record", type=Path,
                   help="With --video: write a scenario file instead of scoring")
    p.add_argument("--interval", type=float, default=1.5,
                   help="Sampling interval in seconds (default: 1.5)")
    p.add_argument("--flicker", type=float, default=0.0,
                   help="With --synthetic: drop this fraction of detections (0-1)")
    p.add_argument("--repeat", type=int, default=3,
                   help="With --synthetic: repeat the block N times for finer "
                        "FAR/FRR resolution (default: 3)")

    p.add_argument("--model", type=Path, default=None, help="LBPH model for --video")
    p.add_argument("--threshold", type=float, default=65.0,
                   help="Recognition threshold for --video (default: 65)")
    p.add_argument("--yunet", action="store_true", help="Use YuNet for --video")

    cf = p.add_argument_group("state machine config")
    cf.add_argument("--mode", choices=[m.value for m in Mode],
                    default=Mode.SECURITY.value)
    cf.add_argument("--delay", type=float, default=10.0)
    cf.add_argument("--max-body-only", type=float, default=20.0)
    cf.add_argument("--max-without-face", type=float, default=90.0)
    cf.add_argument("--intruder-count", type=int, default=2)
    cf.add_argument("--intruder-window", type=float, default=6.0)
    cf.add_argument("--cooldown", type=float, default=30.0)
    cf.add_argument("--startup-grace", type=float, default=5.0)
    cf.add_argument("--anti-spoof-timeout", type=float, default=0.0)

    out = p.add_argument_group("output")
    out.add_argument("--far-window", type=float, default=5.0,
                     help="Seconds after an intruder appears that still count "
                          "as a catch (default: 5)")
    out.add_argument("--json", type=Path, help="Write the report as JSON")
    out.add_argument("--sweep", type=str, default=None,
                     help="Re-score across values, e.g. 'delay=5,10,20'")
    out.add_argument("--fail-if-far-above", type=float, default=None,
                     help="Exit 1 if FAR exceeds this fraction (CI gate)")
    out.add_argument("--fail-if-frr-above", type=float, default=None,
                     help="Exit 1 if FRR exceeds this fraction (CI gate)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chosen = [bool(args.scenario), bool(args.video), bool(args.synthetic)]
    if sum(chosen) != 1:
        print("ERROR: pick exactly one of --scenario / --video / --synthetic",
              file=sys.stderr)
        return 2

    cfg = Config(
        absence_delay=args.delay,
        max_body_only=args.max_body_only,
        max_without_face=args.max_without_face,
        intruder_count=args.intruder_count,
        intruder_window=args.intruder_window,
        cooldown=args.cooldown,
        startup_grace=args.startup_grace,
        anti_spoof_timeout=args.anti_spoof_timeout,
        mode=Mode(args.mode),
    )

    # ── gather observations + labels ────────────────────────────────────
    if args.synthetic:
        observations, labels = synth_scenario(args.interval, args.flicker,
                                             repeat=args.repeat)
        print(f"synthetic scenario: {len(observations)} ticks, "
              f"{len(labels)} labelled intervals, flicker={args.flicker}")
    elif args.video:
        if not args.video.exists():
            print(f"ERROR: no such video: {args.video}", file=sys.stderr)
            return 2
        stream = observations_from_video(
            args.video, args.interval, args.model, args.threshold, args.yunet)
        if args.record:
            n = write_scenario(args.record, stream)
            print(f"wrote {n} observations to {args.record}")
            print("now iterate without re-decoding:  "
                  f"lock-on-absence-replay --scenario {args.record} --labels ...")
            return 0
        observations = list(stream)
        if not args.labels:
            print("ERROR: --labels is required to score (or use --record)",
                  file=sys.stderr)
            return 2
        labels = load_labels(args.labels)
    else:
        observations = list(read_scenario(args.scenario))
        if not args.labels:
            print("ERROR: --labels is required with --scenario", file=sys.stderr)
            return 2
        labels = load_labels(args.labels)

    if not observations:
        print("ERROR: no observations to replay", file=sys.stderr)
        return 2

    report = replay(observations, labels, cfg, args.far_window)
    print(report.render())

    if args.sweep:
        try:
            field_name, raw = args.sweep.split("=", 1)
            values = [float(v) for v in raw.split(",")]
        except ValueError:
            print("ERROR: --sweep must look like 'delay=5,10,20'", file=sys.stderr)
            return 2
        if field_name not in Config.__dataclass_fields__:
            print(f"ERROR: unknown config field {field_name!r}", file=sys.stderr)
            return 2
        print(sweep(observations, labels, cfg, field_name, values, args.far_window))

    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"report written to {args.json}")

    rc = 0
    if (args.fail_if_far_above is not None and report.far is not None
            and report.far > args.fail_if_far_above):
        print(f"GATE FAILED: FAR {report.far:.3f} > {args.fail_if_far_above}",
              file=sys.stderr)
        rc = 1
    if (args.fail_if_frr_above is not None and report.frr is not None
            and report.frr > args.fail_if_frr_above):
        print(f"GATE FAILED: FRR {report.frr:.3f} > {args.fail_if_frr_above}",
              file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
