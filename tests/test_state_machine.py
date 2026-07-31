"""
Tests for PresenceStateMachine.

These run without a camera, without OpenCV and without sleeping, because every
timer in the machine reads obs.t. That is the property that makes the decision
logic testable at all — protect it.

Coverage intent, in priority order:
  * every Decision/Reason is reachable (no dead branches)
  * the evasions found in review are closed and stay closed
  * cooldown never enables keep-awake while the screen is locked
  * mode actually changes behaviour instead of being a decorative field
"""

from __future__ import annotations

import pytest

from lock_on_absence.state_machine import (
    Config,
    Decision,
    Mode,
    Observation,
    PresenceStateMachine,
    Reason,
    State,
)


def obs(t, *, faces=0, owner=False, scene=False, cam=True, busy=False,
        rec=True, center=None, width=0.0):
    return Observation(t=t, faces=faces, owner_recognized=owner,
                       scene_unchanged=scene, camera_ok=cam, camera_busy=busy,
                       has_recognizer=rec, face_center=center, face_width=width)


def run(psm, st, seq):
    """Feed a sequence, return the list of verdicts."""
    return [psm.step(o, st) for o in seq]


# ═══════════════════════════════════════════════════════════════════════
#  Happy path
# ═══════════════════════════════════════════════════════════════════════

def test_owner_keeps_and_enables_keep_awake():
    psm, st = PresenceStateMachine(Config()), State()
    v = psm.step(obs(1.0, faces=1, owner=True), st)
    assert v.decision is Decision.KEEP
    assert v.keep_awake is True
    assert st.last_proof_of_presence == 1.0


def test_owner_message_is_emitted_once_not_every_tick():
    """The machine owns one-shot logging; the adapter has no dedup variables."""
    psm, st = PresenceStateMachine(Config()), State()
    msgs = [psm.step(obs(t, faces=1, owner=True), st).message
            for t in (1.0, 2.5, 4.0, 5.5)]
    assert msgs[0] is not None
    assert msgs[1:] == [None, None, None]


# ═══════════════════════════════════════════════════════════════════════
#  Absence
# ═══════════════════════════════════════════════════════════════════════

def test_absence_warns_then_locks():
    psm, st = PresenceStateMachine(Config(absence_delay=10.0)), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    assert psm.step(obs(2.0), st).decision is Decision.WARN
    v = psm.step(obs(13.0), st)
    assert v.decision is Decision.LOCK and v.reason is Reason.ABSENCE
    assert v.keep_awake is False
    assert v.detail["absence_seconds"] == pytest.approx(11.0, abs=0.1)


def test_owner_return_resets_absence_timer():
    psm, st = PresenceStateMachine(Config(absence_delay=10.0)), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    psm.step(obs(5.0), st)
    psm.step(obs(6.0, faces=1, owner=True), st)
    assert st.absence_start is None
    assert psm.step(obs(12.0), st).decision is Decision.WARN


# ═══════════════════════════════════════════════════════════════════════
#  Intruder — the flicker evasion
# ═══════════════════════════════════════════════════════════════════════

def test_intruder_locks_after_configured_hits():
    cfg = Config(intruder_count=2, intruder_window=6.0, startup_grace=0.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    assert psm.step(obs(1.0, faces=1), st).decision is Decision.WARN
    v = psm.step(obs(2.5, faces=1), st)
    assert v.decision is Decision.LOCK and v.reason is Reason.INTRUDER


def test_detection_flicker_does_not_defeat_intruder_lock():
    """
    Regression: the old consecutive-streak counter was reset by any zero-face
    frame, so an intruder whose detection flickered was never locked out.
    A sliding window must still fire.
    """
    cfg = Config(intruder_count=2, intruder_window=6.0, startup_grace=0.0,
                 absence_delay=999.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    v1 = psm.step(obs(1.0, faces=1), st)          # seen
    v2 = psm.step(obs(2.0), st)                    # flicker: lost
    v3 = psm.step(obs(3.0, faces=1), st)           # seen again
    assert v1.decision is Decision.WARN
    assert v2.decision is not Decision.LOCK
    assert v3.decision is Decision.LOCK and v3.reason is Reason.INTRUDER


def test_intruder_hits_expire_outside_window():
    cfg = Config(intruder_count=2, intruder_window=5.0, startup_grace=0.0,
                 absence_delay=999.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    psm.step(obs(1.0, faces=1), st)
    v = psm.step(obs(30.0, faces=1), st)   # first hit long expired
    assert v.decision is Decision.WARN


def test_startup_grace_suppresses_intruder_but_not_forever():
    cfg = Config(intruder_count=1, startup_grace=5.0, absence_delay=999.0)
    psm, st = PresenceStateMachine(cfg), State()
    assert psm.step(obs(0.5, faces=1), st).reason is Reason.STARTUP_GRACE
    v = psm.step(obs(6.0, faces=1), st)
    assert v.decision is Decision.LOCK and v.reason is Reason.INTRUDER


# ═══════════════════════════════════════════════════════════════════════
#  Body-only
# ═══════════════════════════════════════════════════════════════════════

def test_body_only_holds_then_times_out():
    cfg = Config(max_body_only=20.0, max_without_face=999.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    v = psm.step(obs(5.0, scene=True), st)
    assert v.decision is Decision.KEEP and v.keep_awake is True
    v = psm.step(obs(40.0, scene=True), st)
    assert v.decision is Decision.LOCK and v.reason is Reason.BODY_TIMEOUT


def test_hard_ceiling_beats_repeated_body_window_refresh():
    """
    Regression: body_only_start was cleared whenever a face flickered back,
    so the body window restarted forever. last_proof_of_presence must impose
    an absolute ceiling.
    """
    cfg = Config(max_body_only=20.0, max_without_face=60.0, cooldown=0.0,
                 intruder_count=99)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    t, last = 1.0, None
    while t < 200.0:
        # Alternate body-only with a non-owner face, which used to reset things.
        o = obs(t, scene=True) if int(t) % 2 else obs(t, faces=1, scene=True)
        last = psm.step(o, st)
        if last.decision is Decision.LOCK:
            break
        t += 5.0
    assert last.decision is Decision.LOCK
    assert t <= 100.0, "ceiling never fired; body-only held the screen too long"


def test_body_only_ignored_in_any_face_mode():
    cfg = Config(absence_delay=10.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True, rec=False), st)
    psm.step(obs(20.0, scene=True, rec=False), st)      # starts the absence timer
    v = psm.step(obs(35.0, scene=True, rec=False), st)  # scene_unchanged is ignored
    assert v.decision is Decision.LOCK and v.reason is Reason.ABSENCE


# ═══════════════════════════════════════════════════════════════════════
#  Camera failure and meeting pause
# ═══════════════════════════════════════════════════════════════════════

def test_camera_failure_fails_closed_in_security_mode():
    cfg = Config(camera_fail_grace=20.0, camera_busy_after=99,
                 mode=Mode.SECURITY)
    psm, st = PresenceStateMachine(cfg), State()
    assert psm.step(obs(1.0, cam=False), st).decision is Decision.WARN
    v = psm.step(obs(25.0, cam=False), st)
    assert v.decision is Decision.LOCK and v.reason is Reason.CAMERA_FAILURE


def test_convenience_mode_warns_instead_of_locking():
    """mode must actually change behaviour, not just sit in the dataclass."""
    cfg = Config(camera_fail_grace=20.0, camera_busy_after=99,
                 mode=Mode.CONVENIENCE)
    assert cfg.on_camera_failure == "warn"
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(1.0, cam=False), st)
    v = psm.step(obs(25.0, cam=False), st)
    assert v.decision is Decision.WARN and v.reason is Reason.CAMERA_FAILURE


def test_security_mode_forces_fail_closed_even_if_asked_otherwise():
    cfg = Config(mode=Mode.SECURITY, on_camera_failure="warn")
    assert cfg.on_camera_failure == "lock"


def test_busy_camera_pauses_rather_than_locking():
    """
    Regression: --meeting-pause was dead code, so a Teams call held the camera
    and the fail-closed path locked the machine every ~20s.
    """
    cfg = Config(meeting_pause=30.0, camera_fail_grace=20.0)
    psm, st = PresenceStateMachine(cfg), State()
    v = psm.step(obs(1.0, cam=False, busy=True), st)
    assert v.decision is Decision.PAUSE and v.reason is Reason.CAMERA_BUSY
    for t in (2.0, 10.0, 25.0):
        assert psm.step(obs(t, cam=False, busy=True), st).decision is Decision.PAUSE


def test_pause_budget_is_capped_so_it_cannot_hide_forever():
    cfg = Config(meeting_pause=30.0, meeting_pause_max=60.0,
                 camera_fail_grace=5.0, camera_busy_after=1)
    psm, st = PresenceStateMachine(cfg), State()
    t, decisions = 0.0, []
    for _ in range(40):
        decisions.append(psm.step(obs(t, cam=False, busy=True), st).decision)
        t += 15.0
    assert Decision.LOCK in decisions, "pause budget never expired"


def test_camera_recovery_clears_failure_state():
    cfg = Config(camera_fail_grace=20.0, camera_busy_after=99)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(1.0, cam=False), st)
    assert st.first_camera_fail is not None
    psm.step(obs(3.0, faces=1, owner=True), st)
    assert st.first_camera_fail is None
    assert st.camera_fail_streak == 0


# ═══════════════════════════════════════════════════════════════════════
#  Cooldown
# ═══════════════════════════════════════════════════════════════════════

def test_cooldown_suppresses_new_decisions():
    cfg = Config(cooldown=30.0, absence_delay=1.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    psm.step(obs(5.0), st)                              # absence timer starts
    assert psm.step(obs(7.0), st).decision is Decision.LOCK
    v = psm.step(obs(10.0, faces=1), st)
    assert v.decision is Decision.KEEP and v.reason is Reason.COOLDOWN


def test_cooldown_never_enables_keep_awake():
    """
    Regression: KEEP was returned during cooldown and the adapter read it as
    'unlocked', so the agent suppressed OS sleep while the screen was locked.
    """
    cfg = Config(cooldown=30.0, absence_delay=1.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    psm.step(obs(5.0), st)
    assert psm.step(obs(7.0), st).decision is Decision.LOCK
    for t in (8.0, 15.0, 30.0):
        assert psm.step(obs(t, faces=1, owner=True), st).keep_awake is False


def test_decisions_resume_after_cooldown():
    cfg = Config(cooldown=10.0, absence_delay=1.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    psm.step(obs(5.0), st)
    assert psm.step(obs(7.0), st).decision is Decision.LOCK   # inside cooldown now
    psm.step(obs(30.0), st)                                   # cooldown expired
    assert psm.step(obs(45.0), st).decision is Decision.LOCK


# ═══════════════════════════════════════════════════════════════════════
#  Anti-spoof (weak heuristic, off by default)
# ═══════════════════════════════════════════════════════════════════════

def test_anti_spoof_disabled_by_default():
    psm, st = PresenceStateMachine(Config()), State()
    for t in range(0, 200, 2):
        v = psm.step(obs(float(t), faces=1, owner=True,
                         center=(100.0, 100.0), width=200.0), st)
        assert v.decision is not Decision.LOCK


def test_anti_spoof_fires_on_perfectly_static_face_when_enabled():
    cfg = Config(anti_spoof_timeout=15.0)
    psm, st = PresenceStateMachine(cfg), State()
    last = None
    for t in range(0, 60, 2):
        last = psm.step(obs(float(t), faces=1, owner=True,
                            center=(100.0, 100.0), width=200.0), st)
        if last.decision is Decision.LOCK:
            break
    assert last.decision is Decision.LOCK and last.reason is Reason.SPOOF


def test_anti_spoof_resets_when_face_moves():
    cfg = Config(anti_spoof_timeout=15.0)
    psm, st = PresenceStateMachine(cfg), State()
    for i, t in enumerate(range(0, 120, 2)):
        v = psm.step(obs(float(t), faces=1, owner=True,
                         center=(100.0 + i * 20, 100.0), width=200.0), st)
        assert v.decision is not Decision.LOCK


# ═══════════════════════════════════════════════════════════════════════
#  Invariants
# ═══════════════════════════════════════════════════════════════════════

def test_every_reason_is_reachable():
    """A Reason that no code path returns is dead vocabulary. Catch it here."""
    seen: set[Reason] = set()

    def collect(cfg, seq):
        psm, st = PresenceStateMachine(cfg), State()
        for o in seq:
            seen.add(psm.step(o, st).reason)

    collect(Config(), [obs(0.0, faces=1, owner=True), obs(20.0)])
    collect(Config(intruder_count=1, startup_grace=0.0),
            [obs(0.0, faces=1, owner=True), obs(1.0, faces=1)])
    collect(Config(intruder_count=1, startup_grace=10.0), [obs(0.5, faces=1)])
    collect(Config(max_body_only=5.0),
            [obs(0.0, faces=1, owner=True), obs(10.0, scene=True),
             obs(30.0, scene=True)])
    collect(Config(camera_busy_after=99, camera_fail_grace=1.0),
            [obs(0.0, cam=False), obs(10.0, cam=False)])
    collect(Config(camera_busy_after=1), [obs(0.0, cam=False, busy=True)])
    collect(Config(cooldown=30.0, absence_delay=1.0),
            [obs(0.0, faces=1, owner=True), obs(5.0), obs(6.0, faces=1, owner=True)])
    collect(Config(anti_spoof_timeout=5.0),
            [obs(float(t), faces=1, owner=True, center=(1.0, 1.0), width=100.0)
             for t in range(0, 40, 2)])

    missing = set(Reason) - seen
    assert not missing, f"unreachable Reason values: {sorted(r.value for r in missing)}"


def test_observation_rejects_incoherent_input():
    with pytest.raises(ValueError):
        Observation(t=0.0, faces=0, owner_recognized=True)
    with pytest.raises(ValueError):
        Observation(t=0.0, faces=-1)


def test_lock_always_clears_transient_timers():
    cfg = Config(absence_delay=1.0)
    psm, st = PresenceStateMachine(cfg), State()
    psm.step(obs(0.0, faces=1, owner=True), st)
    psm.step(obs(1.0, scene=True), st)
    psm.step(obs(20.0), st)                 # absence timer starts here
    v = psm.step(obs(40.0), st)
    assert v.decision is Decision.LOCK
    assert st.absence_start is None
    assert st.body_only_start is None
    assert st.intruder_hits == []
    assert st.static_since is None


def test_machine_is_deterministic():
    """Same inputs twice, byte-identical verdicts. Replay depends on this."""
    seq = [obs(float(t), faces=t % 3, owner=(t % 3 == 1), scene=(t % 2 == 0))
           for t in range(0, 90, 3)]
    seq = [o for o in seq if not (o.owner_recognized and o.faces == 0)]

    def once():
        psm, st = PresenceStateMachine(Config()), State()
        return [(v.decision, v.reason) for v in run(psm, st, seq)]

    assert once() == once()


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        Config(on_camera_failure="maybe")
