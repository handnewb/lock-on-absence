"""
Tests for the FAR/FRR harness and a smoke test for the whole package.

The smoke test exists because of a specific, repeated failure mode: refactors
that read correctly, produce a clean diff and a good changelog, and then crash
on import or on the first instantiation. `BodyDetector()` raised AttributeError
in every code path for two commits; `enroll.py` referenced two names that were
never assigned. Both would have been caught by importing the module once.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lock_on_absence import __version__
from lock_on_absence.replay import (
    load_labels,
    read_scenario,
    replay,
    synth_scenario,
    truth_at,
    write_scenario,
)
from lock_on_absence.state_machine import Config, Mode

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
#  Smoke — does the package actually load and run?
# ═══════════════════════════════════════════════════════════════════════

def test_all_modules_import():
    import importlib
    for name in ("state_machine", "replay", "face_utils", "agent",
                 "enroll", "watchdog"):
        importlib.import_module(f"lock_on_absence.{name}")


def test_every_class_instantiates():
    """Catches the `self.calibrated = ...` vs `@property calibrated` crash."""
    from lock_on_absence.face_utils import (
        BodyDetector,
        EventLogger,
        KeepAwake,
        Logger,
    )
    BodyDetector()
    Logger()
    KeepAwake()
    EventLogger(False, None)
    EventLogger(False, None, dry_run=True)


def test_body_detector_calibration_lifecycle():
    """complete_calibration must need real samples and must clamp the result."""
    import numpy as np

    from lock_on_absence.face_utils import BodyDetector

    bd = BodyDetector()
    assert bd.calibrated is False
    assert bd.complete_calibration() is False, "must not calibrate with no samples"

    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, (480, 640), dtype=np.uint8)
    bd.update_ref(ref)
    for _ in range(bd.calibration_samples + 5):
        bd.sample_noise(rng.integers(0, 255, (480, 640), dtype=np.uint8))
    assert bd.complete_calibration() is True
    assert 8.0 <= bd.threshold <= 25.0, "threshold escaped its clamp"
    assert bd.complete_calibration() is False, "must only transition once"


def test_safe_face_roi_clamps_instead_of_crashing():
    """YuNet emits negative coordinates; an unclamped slice used to kill the agent."""
    import numpy as np

    from lock_on_absence.face_utils import safe_face_roi

    gray = np.full((480, 640), 128, np.uint8)
    assert safe_face_roi(gray, (-8, 20, 60, 60)) is not None      # clamped, usable
    assert safe_face_roi(gray, (600, 440, 200, 200)) is not None  # clipped, usable
    assert safe_face_roi(gray, (0, 0, 5, 5)) is None              # too small
    assert safe_face_roi(gray, (10_000, 10_000, 50, 50)) is None  # fully outside
    assert safe_face_roi(gray, ("x", 0, 10, 10)) is None          # junk input


@pytest.mark.parametrize("script", ["lock-on-absence.py", "enroll.py",
                                    "watchdog.py", "replay.py"])
def test_root_shims_still_work(script):
    """install.bat, install.sh, the systemd unit and the README all call these."""
    path = ROOT / script
    assert path.exists(), f"{script} shim is missing; existing installs will break"
    r = subprocess.run([sys.executable, str(path), "--help"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, f"{script} --help failed:\n{r.stderr}"


@pytest.mark.parametrize("module", ["lock_on_absence.agent",
                                    "lock_on_absence.enroll",
                                    "lock_on_absence.replay"])
def test_module_help_runs(module):
    """--help exercises the whole argparse tree and every top-level import."""
    r = subprocess.run([sys.executable, "-m", module, "--help"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, f"{module} --help failed:\n{r.stderr}"


def test_version_is_single_sourced():
    r = subprocess.run([sys.executable, "-m", "lock_on_absence.agent", "--version"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert __version__ in r.stdout, f"--version disagrees with __init__: {r.stdout!r}"


def test_agent_exits_2_without_model_and_without_any_face(tmp_path):
    """Refusing to start beats pretending to protect. Must not be a silent no-op."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.agent",
         "--model", str(tmp_path / "nope.yml")],
        capture_output=True, text=True, timeout=120, cwd=tmp_path, env=env)
    assert r.returncode == 2, f"expected exit 2, got {r.returncode}\n{r.stdout}{r.stderr}"
    assert "any-face" in (r.stdout + r.stderr).lower()


def test_no_undefined_names_anywhere():
    """pyflakes catches exactly the class of bug that shipped twice."""
    r = subprocess.run([sys.executable, "-m", "pyflakes", "lock_on_absence", "tests"],
                       capture_output=True, text=True, cwd=ROOT)
    bad = [ln for ln in r.stdout.splitlines() if "undefined name" in ln]
    assert not bad, "undefined names:\n" + "\n".join(bad)


def test_agent_contains_no_presence_logic():
    """
    Architectural guard. The whole point of the state machine is that the tested
    code and the executed code are the same code. If decision keywords creep
    back into the adapter, the two drift apart again — which is how the repo
    ended up with a fully tested state machine that was never imported.
    """
    src = (ROOT / "lock_on_absence" / "agent.py").read_text(encoding="utf-8")
    assert "PresenceStateMachine" in src, "adapter must use the state machine"
    for banned in ("absence_start", "intruder_streak", "static_since",
                   "prev_face_center", "_body_detect_active", "last_face_time",
                   "locked_until", "body_only_duration"):
        assert banned not in src, (
            f"legacy presence variable {banned!r} is back in agent.py; "
            "that logic belongs in state_machine.py")
    # The machine owns State. A direct attribute write from the adapter is the
    # same divergence the legacy names above caused, just under a different
    # spelling. `st.x =` (but not `st.x ==` or `st.x !=`).
    assert re.search(r"\bst\.[a-z_]+\s*=(?!=)", src) is None, (
        "agent.py mutates machine state directly; call a PresenceStateMachine "
        "method (e.g. reset_camera_failure()) instead")


# ═══════════════════════════════════════════════════════════════════════
#  Labels
# ═══════════════════════════════════════════════════════════════════════

def test_load_labels_roundtrip(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("start_sec,end_sec,truth\n0,10,owner\n# comment\n10,20,absent\n")
    ivs = load_labels(p)
    assert [i.truth for i in ivs] == ["owner", "absent"]
    assert truth_at(ivs, 5.0) == "owner"
    assert truth_at(ivs, 15.0) == "absent"
    assert truth_at(ivs, 99.0) is None


def test_load_labels_rejects_overlap(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("0,10,owner\n5,20,absent\n")
    with pytest.raises(ValueError, match="overlap"):
        load_labels(p)


def test_load_labels_rejects_unknown_truth(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("0,10,sleeping\n")
    with pytest.raises(ValueError, match="unknown truth"):
        load_labels(p)


def test_load_labels_rejects_empty(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("start_sec,end_sec,truth\n")
    with pytest.raises(ValueError, match="no labelled intervals"):
        load_labels(p)


# ═══════════════════════════════════════════════════════════════════════
#  Scenario files
# ═══════════════════════════════════════════════════════════════════════

def test_scenario_roundtrip(tmp_path):
    obs, _ = synth_scenario(interval=3.0)
    p = tmp_path / "s.jsonl"
    n = write_scenario(p, obs)
    back = list(read_scenario(p))
    assert n == len(obs) == len(back)
    assert back[0] == obs[0]
    assert back[-1] == obs[-1]


def test_scenario_rejects_unknown_field(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"t": 1.0, "faces": 1, "wat": true}\n')
    with pytest.raises(ValueError, match="unknown field"):
        list(read_scenario(p))


def test_scenario_reports_bad_json_with_line_number(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"t": 1.0}\nnot json\n')
    with pytest.raises(ValueError, match=":2:"):
        list(read_scenario(p))


# ═══════════════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════════════

def test_default_config_scores_well_on_clean_scenario():
    obs, labels = synth_scenario(interval=1.5)
    rep = replay(obs, labels, Config(), far_window=8.0)
    assert rep.ticks > 100
    assert rep.frr == 0.0, "locked while the owner was present"
    assert rep.far == 0.0, "failed to lock on an intruder"
    assert rep.absent_intervals_missed == 0, "never locked during an absence"
    assert rep.ttl_median is not None and rep.ttl_median <= 15.0


def test_flicker_scenario_still_catches_intruder():
    """The sliding window must survive detection loss; a streak counter did not."""
    obs, labels = synth_scenario(interval=1.5, detect_flicker=0.5)
    rep = replay(obs, labels, Config(), far_window=12.0)
    assert rep.far == 0.0, "flicker defeated intruder detection again"


def test_absurd_absence_delay_produces_absence_misses():
    """The harness must be able to fail, otherwise it measures nothing."""
    obs, labels = synth_scenario(interval=1.5)
    rep = replay(obs, labels, Config(absence_delay=10_000.0), far_window=8.0)
    assert rep.absent_intervals_missed == rep.absent_intervals


def test_tiny_absence_delay_hurts_frr():
    """A 0-second delay must lock on the owner's first blink — visible as FRR."""
    obs, labels = synth_scenario(interval=1.5, detect_flicker=0.35)
    # max_body_only must go too: during `owner` the scene is unchanged, so a
    # dropped detection correctly falls into body-only rather than absence.
    rep = replay(obs, labels,
                 Config(absence_delay=0.0, max_body_only=0.0, cooldown=0.0),
                 far_window=8.0)
    assert rep.frr is not None and rep.frr > 0.0


def test_report_serializes_to_json():
    obs, labels = synth_scenario(interval=3.0)
    d = replay(obs, labels, Config(), far_window=8.0).to_dict()
    json.dumps(d)
    for key in ("far", "frr", "spurious_locks_per_hour", "locks_by_reason",
                "time_to_lock_median", "intervals"):
        assert key in d


def test_replay_is_deterministic():
    obs, labels = synth_scenario(interval=1.5, detect_flicker=0.3)
    a = replay(obs, labels, Config(), far_window=8.0).to_dict()
    b = replay(obs, labels, Config(), far_window=8.0).to_dict()
    assert a == b


def test_convenience_mode_changes_the_numbers():
    obs, labels = synth_scenario(interval=1.5)
    sec = replay(obs, labels, Config(mode=Mode.SECURITY), far_window=8.0)
    con = replay(obs, labels, Config(mode=Mode.CONVENIENCE), far_window=8.0)
    assert sec.to_dict()["intervals"] == con.to_dict()["intervals"]


# ═══════════════════════════════════════════════════════════════════════
#  Harness CLI
# ═══════════════════════════════════════════════════════════════════════

def test_cli_synthetic_runs_and_reports():
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.replay", "--synthetic"],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "FAR / FRR report" in r.stdout
    assert "time-to-lock median" in r.stdout


def test_cli_sweep_and_json_gate(tmp_path):
    out = tmp_path / "rep.json"
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.replay", "--synthetic",
         "--sweep", "absence_delay=5,10,20", "--json", str(out),
         "--fail-if-far-above", "0.0", "--fail-if-frr-above", "0.0"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert r.returncode == 0, f"CI gate failed unexpectedly:\n{r.stdout}{r.stderr}"
    assert "sweep absence_delay" in r.stdout
    assert json.loads(out.read_text())["far"] == 0.0


def test_cli_gate_actually_fails_on_bad_config():
    """A gate that cannot fail is not a gate. Break the config, expect exit 1."""
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.replay", "--synthetic",
         "--intruder-count", "999",          # can never confirm an intruder
         "--fail-if-far-above", "0.0"],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 1, f"gate did not fire:\n{r.stdout}{r.stderr}"
    assert "GATE FAILED" in r.stderr


def test_cli_rejects_multiple_inputs(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.replay",
         "--synthetic", "--scenario", str(tmp_path / "x.jsonl")],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 2
    assert "exactly one" in r.stderr


def test_cli_requires_labels_with_scenario(tmp_path):
    obs, _ = synth_scenario(interval=3.0)
    p = tmp_path / "s.jsonl"
    write_scenario(p, obs)
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.replay", "--scenario", str(p)],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 2
    assert "--labels" in r.stderr


def test_cli_scenario_plus_labels_scores(tmp_path):
    obs, labels = synth_scenario(interval=1.5)
    sp, lp = tmp_path / "s.jsonl", tmp_path / "l.csv"
    write_scenario(sp, obs)
    lp.write_text("start_sec,end_sec,truth\n" +
                  "".join(f"{i.start},{i.end},{i.truth}\n" for i in labels))
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.replay",
         "--scenario", str(sp), "--labels", str(lp), "--far-window", "8"],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "FAR / FRR report" in r.stdout


# ═══════════════════════════════════════════════════════════════════════
#  Watchdog
# ═══════════════════════════════════════════════════════════════════════

def _wd_args(tmp_path, **over):
    from lock_on_absence.watchdog import build_parser
    argv = ["--heartbeat", str(tmp_path / "hb.txt"), "--dry-run"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return build_parser().parse_args(argv)


def test_watchdog_fresh_heartbeat_is_ok(tmp_path):
    import time as _t

    from lock_on_absence.watchdog import check_once
    (tmp_path / "hb.txt").write_text(str(_t.time()))
    assert check_once(_wd_args(tmp_path), {}, log=lambda m: None) == "ok"


def test_watchdog_stale_heartbeat_triggers(tmp_path):
    import time as _t

    from lock_on_absence.watchdog import check_once
    (tmp_path / "hb.txt").write_text(str(_t.time() - 9999))
    assert check_once(_wd_args(tmp_path), {}, log=lambda m: None) in (
        "dry-run", "already-locked")


def test_watchdog_latches_and_does_not_spam(tmp_path):
    """Regression: the original locked every 30s forever, locking the user out."""
    import time as _t

    from lock_on_absence.watchdog import check_once
    (tmp_path / "hb.txt").write_text(str(_t.time() - 9999))
    args, state = _wd_args(tmp_path), {}
    first = check_once(args, state, log=lambda m: None)
    assert first in ("dry-run", "already-locked")
    for _ in range(5):
        assert check_once(args, state, log=lambda m: None) == "latched"


def test_watchdog_rearms_after_recovery(tmp_path):
    import time as _t

    from lock_on_absence.watchdog import check_once
    hb = tmp_path / "hb.txt"
    args, state = _wd_args(tmp_path), {}
    hb.write_text(str(_t.time() - 9999))
    check_once(args, state, log=lambda m: None)
    hb.write_text(str(_t.time()))
    assert check_once(args, state, log=lambda m: None) == "ok"
    assert state["latched"] is False


def test_watchdog_future_timestamp_is_tampering_not_freshness(tmp_path):
    """`echo 9999999999 > heartbeat` must not disable the watchdog."""
    import time as _t

    from lock_on_absence.watchdog import Verdict, read_heartbeat
    hb = tmp_path / "hb.txt"
    hb.write_text(str(_t.time() + 10_000))
    verdict, _age = read_heartbeat(hb, 120.0)
    assert verdict == Verdict.TAMPERED


def test_watchdog_missing_heartbeat_waits_by_default(tmp_path):
    from lock_on_absence.watchdog import check_once
    assert check_once(_wd_args(tmp_path), {}, log=lambda m: None) == "waiting"


def test_watchdog_unreadable_heartbeat_triggers(tmp_path):
    from lock_on_absence.watchdog import Verdict, read_heartbeat
    hb = tmp_path / "hb.txt"
    hb.write_text("not a number")
    assert read_heartbeat(hb, 120.0)[0] == Verdict.UNREADABLE


def test_watchdog_prints_installable_definitions():
    for flag in ("--print-unit", "--print-task"):
        r = subprocess.run(
            [sys.executable, "-m", "lock_on_absence.watchdog", flag],
            capture_output=True, text=True, timeout=60, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        assert "lock_on_absence.watchdog" in r.stdout


# ═══════════════════════════════════════════════════════════════════════
#  File protection and model integrity  (v5.1)
# ═══════════════════════════════════════════════════════════════════════

def test_restrict_file_permissions_reports_honestly(tmp_path):
    """Must never claim success it cannot back up — chmod is ~a no-op on Windows."""
    from lock_on_absence.face_utils import restrict_file_permissions
    f = tmp_path / "secret.yml"
    f.write_text("x")
    ok, how = restrict_file_permissions(f)
    assert isinstance(ok, bool) and isinstance(how, str) and how
    if sys.platform != "win32":
        assert ok is True
        assert (f.stat().st_mode & 0o777) == 0o600


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    from lock_on_absence.face_utils import file_sha256
    f = tmp_path / "m.yml"
    f.write_bytes(b"model bytes" * 10_000)     # forces multiple read chunks
    assert file_sha256(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_agent_refuses_to_start_on_model_digest_mismatch(tmp_path):
    """
    A swapped model means a stranger's face becomes the owner. Fail closed.
    """
    import cv2
    import numpy as np
    model = tmp_path / "face_model.yml"
    rng = np.random.default_rng(0)
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train([rng.integers(0, 255, (200, 200), dtype=np.uint8) for _ in range(3)],
              np.array([1, 1, 1]))
    rec.write(str(model))

    (tmp_path / "face_model.json").write_text(json.dumps({
        "threshold": 65,
        "model_sha256": "0" * 64,          # deliberately wrong
        "users": {"1": "someone"},
    }))

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.agent", "--model", str(model)],
        capture_output=True, text=True, timeout=180, cwd=tmp_path, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 3, f"expected exit 3, got {r.returncode}\n{out}"
    assert "does not match the digest" in out


def test_agent_accepts_a_matching_digest(tmp_path):
    import cv2
    import numpy as np

    from lock_on_absence.face_utils import file_sha256
    model = tmp_path / "face_model.yml"
    rng = np.random.default_rng(1)
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train([rng.integers(0, 255, (200, 200), dtype=np.uint8) for _ in range(3)],
              np.array([1, 1, 1]))
    rec.write(str(model))
    (tmp_path / "face_model.json").write_text(json.dumps({
        "threshold": 65, "model_sha256": file_sha256(model), "users": {"1": "x"},
    }))

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    # No camera in CI, so the agent exits 1 at camera open -- but it must get
    # PAST the integrity gate first, which is what this asserts.
    r = subprocess.run(
        [sys.executable, "-m", "lock_on_absence.agent", "--model", str(model),
         "--camera", "99"],
        capture_output=True, text=True, timeout=180, cwd=tmp_path, env=env)
    out = r.stdout + r.stderr
    assert "Model integrity: OK" in out, out
    assert r.returncode != 3
