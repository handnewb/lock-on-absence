"""
PresenceStateMachine — pure decision logic (no OpenCV, no OS).

Extracted per adversarial review §4.6: every lock decision is a
deterministic function of (observation, config).  Testable with pytest.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Decision(Enum):
    KEEP = auto()       # screen stays unlocked
    LOCK = auto()       # lock the screen
    WARN = auto()       # log a warning, don't lock yet


class Reason(Enum):
    INTRUDER = "intruder"                    # non-owner face detected
    ABSENCE = "absence"                      # no face + no body → timer expired
    BODY_TIMEOUT = "body_timeout"            # body-only detection exceeded grace
    SPOOF = "spoof"                          # movement anti-spoof triggered
    CAMERA_FAILURE = "camera_failure"        # fail-closed: no camera signal
    NONE = "none"                            # no decision needed


@dataclass
class Observation:
    """One frame's worth of sensor data, reduced to decision inputs."""
    t: float                          # timestamp (seconds since epoch)
    faces: int                        # number of faces detected (0+)
    owner_recognized: bool            # at least one face is recognized owner
    scene_unchanged: bool             # body/scene detection says "same as ref"
    camera_ok: bool                   # camera read succeeded


@dataclass
class Config:
    """Tunable parameters.  All times in seconds."""
    absence_delay: float = 10.0       # seconds without owner before lock
    max_body_only: float = 20.0       # max seconds with body-only detection
    intruder_streak: int = 2          # consecutive non-owner frames to lock
    cooldown: float = 30.0            # seconds after lock before new checks
    camera_fail_grace: float = 20.0   # seconds before fail-closed lock
    anti_spoof_timeout: float = 0.0   # 0 = disabled
    mode: str = "security"            # security | convenience


@dataclass
class State:
    """Mutable state tracked across steps."""
    last_owner_time: float = 0.0
    last_proof_of_presence: float = 0.0
    absence_start: Optional[float] = None
    body_only_start: Optional[float] = None
    intruder_count: int = 0
    last_lock_time: float = 0.0
    first_camera_fail: Optional[float] = None


class PresenceStateMachine:
    """
    Pure state machine for presence-based screen locking.

    Usage:
        psm = PresenceStateMachine(Config())
        state = State()
        for obs in observations:
            decision, reason = psm.step(obs, state)
            if decision == Decision.LOCK:
                lock_screen(reason)
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()

    def step(self, obs: Observation, st: State) -> tuple[Decision, Reason]:
        """
        Process one observation. Returns (Decision, Reason).

        Order of evaluation matters — earlier checks short-circuit:
        1. Cooldown after recent lock
        2. Camera failure → fail-closed
        3. Owner recognized → reset all timers, KEEP
        4. Intruder (non-owner face) → streak → LOCK
        5. No face, no body → absence timer → LOCK
        6. No face, body present → body-only timer → LOCK on timeout
        7. Any face (no recognizer) → KEEP
        """
        now = obs.t

        # ── 1. Cooldown ──
        if st.last_lock_time > 0 and now - st.last_lock_time < self.cfg.cooldown:
            return Decision.KEEP, Reason.NONE

        # ── 2. Camera failure (fail-closed) ──
        if not obs.camera_ok:
            if st.first_camera_fail is None:
                st.first_camera_fail = now
            if now - st.first_camera_fail > self.cfg.camera_fail_grace:
                st.last_lock_time = now
                st.first_camera_fail = None
                return Decision.LOCK, Reason.CAMERA_FAILURE
            return Decision.WARN, Reason.NONE

        st.first_camera_fail = None  # camera recovered

        # ── 3. Owner recognized ──
        if obs.owner_recognized and obs.faces > 0:
            st.last_owner_time = now
            st.last_proof_of_presence = now
            st.absence_start = None
            st.body_only_start = None
            st.intruder_count = 0
            return Decision.KEEP, Reason.NONE

        # ── 4. Intruder (face detected, not owner) ──
        if obs.faces > 0 and not obs.owner_recognized:
            st.intruder_count += 1
            if st.intruder_count >= self.cfg.intruder_streak:
                st.last_lock_time = now
                st.intruder_count = 0
                return Decision.LOCK, Reason.INTRUDER
            return Decision.WARN, Reason.NONE

        # ── 5. No face at all ──
        st.intruder_count = 0
        if obs.faces == 0:
            # Body detection: scene unchanged?
            if obs.scene_unchanged:
                if st.body_only_start is None:
                    st.body_only_start = now
                if now - st.body_only_start > self.cfg.max_body_only:
                    st.last_lock_time = now
                    st.body_only_start = None
                    return Decision.LOCK, Reason.BODY_TIMEOUT
                return Decision.KEEP, Reason.NONE

            # No body → absence
            if st.absence_start is None:
                st.absence_start = now
            if now - st.absence_start > self.cfg.absence_delay:
                st.last_lock_time = now
                st.absence_start = None
                return Decision.LOCK, Reason.ABSENCE
            return Decision.WARN, Reason.NONE

        # ── 6. Any face (no recognizer) ──
        return Decision.KEEP, Reason.NONE
