"""
PresenceStateMachine — the *single* source of truth for every lock decision.

Design contract (adversarial review §3.1 / §4.6):

  * This module imports nothing from OpenCV and makes no OS calls.
  * Every decision — including anti-spoof, camera failure, meeting pause,
    keep-awake and the one-shot log messages — is produced HERE.
  * The agent loop is a dumb adapter: build Observation, call step(),
    execute Verdict. It owns no timers and no decision branches.

If you find yourself adding an `if` about presence in agent.py, it belongs
in this file instead. That rule is what keeps the tested code and the
executed code the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════
#  Vocabulary
# ═══════════════════════════════════════════════════════════════════════

class Decision(Enum):
    KEEP = "keep"     # leave the screen unlocked
    WARN = "warn"     # a timer is running; nothing to do yet
    LOCK = "lock"     # lock the screen now
    PAUSE = "pause"   # monitoring suspended (camera owned by another app)


class Reason(Enum):
    NONE = "none"
    INTRUDER = "intruder"                # non-owner face confirmed
    ABSENCE = "absence"                  # no face, no body, timer expired
    BODY_TIMEOUT = "body_timeout"        # body-only held too long without a face
    SPOOF = "spoof"                      # face static far too long (weak heuristic)
    CAMERA_FAILURE = "camera_failure"    # fail-closed: no usable camera signal
    CAMERA_BUSY = "camera_busy"          # another application holds the camera
    COOLDOWN = "cooldown"                # inside post-lock quiet window
    STARTUP_GRACE = "startup_grace"      # suppressed while the user settles in


class Mode(str, Enum):
    SECURITY = "security"
    CONVENIENCE = "convenience"


# ═══════════════════════════════════════════════════════════════════════
#  Inputs
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """
    One sampling tick reduced to decision inputs.

    `t` MUST come from a monotonic clock. Wall-clock time breaks every timer
    here on NTP correction and on laptop suspend/resume.
    """
    t: float
    faces: int = 0                                   # number of faces detected
    owner_recognized: bool = False                   # some face matched an enrolled user
    scene_unchanged: bool = False                    # frame-diff says "same as reference"
    camera_ok: bool = False                          # read() returned a usable frame
    camera_busy: bool = False                        # another app holds the device
    has_recognizer: bool = True                      # False => --any-face mode
    face_center: tuple[float, float] | None = None  # owner face centroid, px
    face_width: float = 0.0                          # owner face width, px

    def __post_init__(self) -> None:
        if self.faces < 0:
            raise ValueError("faces must be >= 0")
        if self.owner_recognized and self.faces == 0:
            raise ValueError("owner_recognized=True requires faces >= 1")


@dataclass
class Config:
    """Tunables. All durations in seconds."""
    absence_delay: float = 10.0        # no face + no body -> lock after this
    max_body_only: float = 20.0        # body-only may hold the screen this long
    max_without_face: float = 90.0     # hard ceiling since last real face proof
    intruder_count: int = 2            # non-owner detections needed to lock
    intruder_window: float = 6.0       # ...within this sliding window
    cooldown: float = 30.0             # quiet period after any lock
    camera_fail_grace: float = 20.0    # no signal this long -> fail-closed lock
    camera_busy_after: int = 5         # consecutive failures before "busy" verdict
    meeting_pause: float = 30.0        # how long to stay paused when busy
    meeting_pause_max: float = 900.0   # absolute cap on cumulative pausing
    startup_grace: float = 5.0         # suppress intruder locks at boot
    anti_spoof_timeout: float = 0.0    # 0 = disabled (recommended; heuristic is weak)
    anti_spoof_min_move: float = 4.0   # px floor for "it moved"
    on_camera_failure: str = "lock"    # lock | warn
    mode: Mode = Mode.SECURITY

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = Mode(self.mode)
        if self.on_camera_failure not in ("lock", "warn"):
            raise ValueError("on_camera_failure must be 'lock' or 'warn'")
        # mode is not decoration: it forces the safety-relevant defaults.
        if self.mode is Mode.SECURITY:
            self.on_camera_failure = "lock"
            self.max_body_only = min(self.max_body_only, 20.0)
            self.max_without_face = min(self.max_without_face, 90.0)
        else:  # CONVENIENCE
            if self.on_camera_failure == "lock":
                self.on_camera_failure = "warn"


@dataclass
class State:
    """Everything the machine remembers. Owned by the machine, not the caller."""
    started_at: float | None = None
    last_proof_of_presence: float | None = None   # last *recognized face*
    absence_start: float | None = None
    body_only_start: float | None = None
    intruder_hits: list[float] = field(default_factory=list)
    last_lock_time: float | None = None
    first_camera_fail: float | None = None
    camera_fail_streak: int = 0
    paused_until: float | None = None
    paused_total: float = 0.0
    prev_face_center: tuple[float, float] | None = None
    static_since: float | None = None
    # Presentation memory, so the machine (not the adapter) owns one-shot logs.
    _phase: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Verdict:
    decision: Decision
    reason: Reason = Reason.NONE
    keep_awake: bool = False           # adapter enables/disables to match this
    message: str | None = None      # emit only when present (transition-only)
    detail: dict = field(default_factory=dict)   # structured payload for SIEM

    @property
    def is_lock(self) -> bool:
        return self.decision is Decision.LOCK


# ═══════════════════════════════════════════════════════════════════════
#  Machine
# ═══════════════════════════════════════════════════════════════════════

class PresenceStateMachine:
    """
    Pure decision function over (Observation, State).

        psm = PresenceStateMachine(Config())
        st = State()
        v = psm.step(obs, st)
        if v.is_lock:
            lock_screen()
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    # ── helpers ───────────────────────────────────────────────────────

    def _phase(self, st: State, name: str, msg: str) -> str | None:
        """Return `msg` only on entering a new phase — kills log spam."""
        if st._phase == name:
            return None
        st._phase = name
        return msg

    def _lock(self, st: State, now: float, reason: Reason,
              msg: str, **detail) -> Verdict:
        st.last_lock_time = now
        st.absence_start = None
        st.body_only_start = None
        st.intruder_hits.clear()
        st.prev_face_center = None
        st.static_since = None
        st.first_camera_fail = None
        st._phase = f"locked:{reason.value}"
        return Verdict(Decision.LOCK, reason, keep_awake=False,
                       message=msg, detail=detail)

    def reset_camera_failure(self, st: State) -> None:
        """Adapter-facing recovery API.

        The machine owns State; after the adapter successfully re-opens the
        camera it calls this instead of poking attributes directly. Doing
        nothing here is safe — step() clears the same fields on the next
        successful read — but explicit recovery keeps the streak from being
        reset by a read that never happened.
        """
        st.camera_fail_streak = 0
        st.first_camera_fail = None

    # ── main entry ────────────────────────────────────────────────────

    def step(self, obs: Observation, st: State) -> Verdict:
        cfg = self.cfg
        now = obs.t

        if st.started_at is None:
            st.started_at = now
        if st.last_proof_of_presence is None:
            # Assume the person who launched it is present, else we lock instantly.
            st.last_proof_of_presence = now

        # ── 0. Paused (camera owned by another app) ────────────────────
        if st.paused_until is not None:
            if now < st.paused_until:
                return Verdict(Decision.PAUSE, Reason.CAMERA_BUSY, keep_awake=False,
                               message=self._phase(st, "paused",
                                                   f"Monitoring paused — camera in use "
                                                   f"({st.paused_until - now:.0f}s left)"),
                               detail={"paused_remaining": round(st.paused_until - now, 1)})
            st.paused_total += cfg.meeting_pause
            st.paused_until = None
            st._phase = ""

        # ── 1. Cooldown after a lock ───────────────────────────────────
        # KEEP here means "do nothing", NOT "unlocked" — keep_awake stays False
        # so we never suppress sleep while the screen is actually locked.
        if st.last_lock_time is not None and now - st.last_lock_time < cfg.cooldown:
            return Verdict(Decision.KEEP, Reason.COOLDOWN, keep_awake=False,
                           message=self._phase(
                               st, "cooldown",
                               f"Cooldown {cfg.cooldown:.0f}s after lock"))

        # ── 2. No usable camera frame ──────────────────────────────────
        if not obs.camera_ok:
            st.camera_fail_streak += 1
            if st.first_camera_fail is None:
                st.first_camera_fail = now

            # 2a. Someone else owns the device -> pause, don't fight it.
            busy = obs.camera_busy or st.camera_fail_streak >= cfg.camera_busy_after
            if busy and st.paused_total < cfg.meeting_pause_max:
                st.paused_until = now + cfg.meeting_pause
                st.first_camera_fail = None
                st.camera_fail_streak = 0
                return Verdict(Decision.PAUSE, Reason.CAMERA_BUSY, keep_awake=False,
                               message=self._phase(
                                   st, "paused",
                                   f"Camera in use by another app — pausing "
                                   f"{cfg.meeting_pause:.0f}s"),
                               detail={"paused_total": round(st.paused_total, 1)})

            # 2b. Broken / unplugged / pause budget exhausted -> fail-closed.
            waited = now - st.first_camera_fail
            if waited > cfg.camera_fail_grace:
                if cfg.on_camera_failure == "lock":
                    return self._lock(st, now, Reason.CAMERA_FAILURE,
                                      f">>> LOCKING (no camera signal for "
                                      f"{waited:.0f}s — fail-closed)",
                                      camera_fail_seconds=round(waited, 1))
                return Verdict(Decision.WARN, Reason.CAMERA_FAILURE, keep_awake=False,
                               message=self._phase(
                                   st, "camfail_warn",
                                   f"WARNING: no camera signal for {waited:.0f}s "
                                   f"(--on-camera-failure warn: not locking)"),
                               detail={"camera_fail_seconds": round(waited, 1)})

            return Verdict(Decision.WARN, Reason.CAMERA_FAILURE, keep_awake=False,
                           message=self._phase(
                               st, "camfail",
                               f"Camera read failed — {cfg.camera_fail_grace:.0f}s "
                               f"until fail-closed lock"),
                           detail={"camera_fail_seconds": round(waited, 1)})

        # Camera is healthy again.
        st.camera_fail_streak = 0
        st.first_camera_fail = None

        # ── 3. Owner present ──────────────────────────────────────────
        # In --any-face mode any detected face counts as the owner. That is
        # insecure by construction and the caller is expected to have said so.
        owner = obs.owner_recognized or (not obs.has_recognizer and obs.faces > 0)

        if owner:
            st.last_proof_of_presence = now
            st.absence_start = None
            st.body_only_start = None
            st.intruder_hits.clear()

            spoof = self._check_static_face(obs, st)
            if spoof is not None:
                return spoof

            return Verdict(Decision.KEEP, Reason.NONE, keep_awake=True,
                           message=self._phase(st, "owner", "Owner present"))

        # Not the owner from here on: no anti-spoof tracking to carry.
        st.prev_face_center = None
        st.static_since = None

        # ── 4. Someone else's face ────────────────────────────────────
        if obs.faces > 0 and obs.has_recognizer:
            # Sliding window, not a consecutive streak: detection flicker used
            # to reset a consecutive counter and defeat intruder locking.
            st.intruder_hits.append(now)
            cutoff = now - cfg.intruder_window
            st.intruder_hits = [h for h in st.intruder_hits if h >= cutoff]

            if now - st.started_at < cfg.startup_grace:
                return Verdict(Decision.WARN, Reason.STARTUP_GRACE, keep_awake=False,
                               message=self._phase(
                                   st, "grace",
                                   f"Startup grace — unrecognized face ignored for "
                                   f"{cfg.startup_grace - (now - st.started_at):.0f}s"))

            if len(st.intruder_hits) >= cfg.intruder_count:
                return self._lock(st, now, Reason.INTRUDER,
                                  ">>> LOCKING (intruder confirmed: "
                                  f"{len(st.intruder_hits)} non-owner detections in "
                                  f"{cfg.intruder_window:.0f}s)",
                                  hits=len(st.intruder_hits))

            return Verdict(Decision.WARN, Reason.INTRUDER, keep_awake=False,
                           message=self._phase(
                               st, "intruder_pending",
                               f"Unrecognized face ({len(st.intruder_hits)}/"
                               f"{cfg.intruder_count}) — watching"))

        # ── 5. No face at all ─────────────────────────────────────────
        # NOTE: intruder_hits is deliberately NOT cleared here. Clearing it on a
        # zero-face frame is exactly what let detection flicker defeat intruder
        # locking: seen -> lost -> seen never accumulated. Let the sliding
        # window expire the hits by time instead. Only owner recognition and an
        # actual lock clear the list.
        since_face = now - st.last_proof_of_presence

        # Hard ceiling: body detection must never hold the screen forever,
        # no matter how often the body-only window got refreshed.
        if obs.has_recognizer and since_face > cfg.max_without_face:
            return self._lock(st, now, Reason.BODY_TIMEOUT,
                              f">>> LOCKING (no verified face for {since_face:.0f}s "
                              f"> hard ceiling {cfg.max_without_face:.0f}s)",
                              seconds_without_face=round(since_face, 1),
                              ceiling=True)

        if obs.scene_unchanged and obs.has_recognizer:
            if st.body_only_start is None:
                st.body_only_start = now
            held = now - st.body_only_start
            if held > cfg.max_body_only:
                return self._lock(st, now, Reason.BODY_TIMEOUT,
                                  f">>> LOCKING (body-only for {held:.0f}s > "
                                  f"{cfg.max_body_only:.0f}s without face re-verification)",
                                  body_only_seconds=round(held, 1))
            return Verdict(Decision.KEEP, Reason.NONE, keep_awake=True,
                           message=self._phase(
                               st, "body",
                               f"No face — body present, re-verify in "
                               f"{cfg.max_body_only - held:.0f}s"),
                           detail={"body_only_seconds": round(held, 1)})

        # ── 6. Genuinely nobody there ─────────────────────────────────
        st.body_only_start = None
        if st.absence_start is None:
            st.absence_start = now
        elapsed = now - st.absence_start
        if elapsed >= cfg.absence_delay:
            return self._lock(st, now, Reason.ABSENCE,
                              ">>> LOCKING (nobody present for "
                              f"{elapsed:.0f}s)",
                              absence_seconds=round(elapsed, 1))

        return Verdict(Decision.WARN, Reason.ABSENCE, keep_awake=False,
                       message=self._phase(
                           st, "absent",
                           f"No face — locking in {cfg.absence_delay - elapsed:.0f}s"),
                       detail={"absence_seconds": round(elapsed, 1)})

    # ── anti-spoof (weak heuristic, off by default) ────────────────────

    def _check_static_face(self, obs: Observation, st: State) -> Verdict | None:
        """
        Lock if the owner's face has not moved at all for a long time.

        Honest assessment: this does NOT detect photo attacks reliably. Haar/DNN
        boxes jitter by several pixels on a static print, and a genuinely still
        human trips it. Disabled by default (anti_spoof_timeout=0) and kept only
        because a very long timeout is a cheap sanity check. Do not describe this
        as liveness detection.
        """
        cfg = self.cfg
        if cfg.anti_spoof_timeout <= 0 or obs.face_center is None:
            return None

        now = obs.t
        threshold = max(obs.face_width * 0.015, cfg.anti_spoof_min_move)
        prev = st.prev_face_center
        st.prev_face_center = obs.face_center

        if prev is None:
            return None

        moved = (abs(obs.face_center[0] - prev[0]) >= threshold
                 or abs(obs.face_center[1] - prev[1]) >= threshold)
        if moved:
            st.static_since = None
            return None

        if st.static_since is None:
            st.static_since = now
            return None

        held = now - st.static_since
        if held > cfg.anti_spoof_timeout:
            return self._lock(st, now, Reason.SPOOF,
                              f">>> LOCKING (face static for {held:.0f}s — "
                              f"possible photo)",
                              static_seconds=round(held, 1))
        return None
