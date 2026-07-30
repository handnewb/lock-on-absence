"""pytest tests for PresenceStateMachine — pure logic, no hardware needed."""
import sys
import pytest
from presence_state_machine import (
    Config, Decision, Observation, PresenceStateMachine, Reason, State,
)


def _obs(t=0.0, faces=0, owner=False, scene=True, camera=True) -> Observation:
    return Observation(t=t, faces=faces, owner_recognized=owner,
                       scene_unchanged=scene, camera_ok=camera)


# ── Owner present → KEEP ──
def test_owner_present_keeps():
    sm = PresenceStateMachine(Config())
    st = State()
    d, r = sm.step(_obs(1.0, faces=1, owner=True), st)
    assert d == Decision.KEEP
    assert st.last_proof_of_presence == 1.0


# ── Intruder → LOCK after streak ──
def test_intruder_streak_locks():
    sm = PresenceStateMachine(Config(intruder_streak=2))
    st = State()
    sm.step(_obs(1.0, faces=1, owner=False), st)     # streak=1
    d, r = sm.step(_obs(2.0, faces=1, owner=False), st)  # streak=2
    assert d == Decision.LOCK
    assert r == Reason.INTRUDER


# ── Absence → LOCK after delay ──
def test_absence_locks_after_delay():
    sm = PresenceStateMachine(Config(absence_delay=5.0))
    st = State()
    d, r = sm.step(_obs(1.0, faces=0, owner=False, scene=False), st)
    assert d == Decision.WARN
    # Advance time past absence_delay
    d, r = sm.step(_obs(7.0, faces=0, owner=False, scene=False), st)
    assert d == Decision.LOCK
    assert r == Reason.ABSENCE


# ── Body-only → KEEP until timeout, then LOCK ──
def test_body_only_timeout():
    sm = PresenceStateMachine(Config(max_body_only=10.0))
    st = State()
    d, r = sm.step(_obs(1.0, faces=0, owner=False, scene=True), st)
    assert d == Decision.KEEP
    d, r = sm.step(_obs(12.0, faces=0, owner=False, scene=True), st)
    assert d == Decision.LOCK
    assert r == Reason.BODY_TIMEOUT


# ── Camera failure → WARN then LOCK ──
def test_camera_failure_fail_closed():
    sm = PresenceStateMachine(Config(camera_fail_grace=5.0))
    st = State()
    d, r = sm.step(_obs(1.0, faces=0, camera=False), st)
    assert d == Decision.WARN
    d, r = sm.step(_obs(7.0, faces=0, camera=False), st)
    assert d == Decision.LOCK
    assert r == Reason.CAMERA_FAILURE


# ── Cooldown blocks locks ──
def test_cooldown_blocks():
    sm = PresenceStateMachine(Config(cooldown=30.0))
    st = State(last_lock_time=10.0)
    d, r = sm.step(_obs(15.0, faces=1, owner=False), st)
    assert d == Decision.KEEP  # cooldown blocks intruder check


# ── Recovery resets everything ──
def test_recovery_resets():
    sm = PresenceStateMachine(Config())
    st = State(absence_start=5.0, intruder_count=1)
    d, r = sm.step(_obs(10.0, faces=1, owner=True), st)
    assert d == Decision.KEEP
    assert st.absence_start is None
    assert st.intruder_count == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
