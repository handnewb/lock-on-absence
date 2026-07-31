# Lock on Absence

> Auto-lock your workstation when you walk away. Facial recognition, a pure-Python decision engine, a FAR/FRR measurement harness, and a SIEM audit trail — no special hardware required.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-Windows%20|%20Linux%20|%20macOS-lightgrey)](https://github.com/handnewb/lock-on-absence)
[![Version](https://img.shields.io/badge/version-5.1.0-brightgreen)](https://github.com/handnewb/lock-on-absence/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/handnewb/lock-on-absence/actions/workflows/ci.yml/badge.svg)](https://github.com/handnewb/lock-on-absence/actions/workflows/ci.yml)

---

## Table of Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [FAR/FRR harness](#farrr-harness)
- [SIEM / event log](#siem--event-log)
- [Decision engine](#decision-engine)
- [Architecture](#architecture)
- [Security model](#security-model)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## How it works

```
┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
│  Webcam  │──▶│  Detector    │──▶│ Recognizer │──▶│     PSM      │──▶ LOCK / KEEP / PAUSE
│ (Haar /  │   │ (Haar /      │   │ (LBPH /    │   │ (pure Python │       │
│  YuNet)  │   │  YuNet)      │   │  SFace*)   │   │  state       │       ▼
└──────────┘   └──────────────┘   └────────────┘   │  machine)    │  ┌──────────────┐
                                                   └──────────────┘  │  SIEM Output │──▶ lock-events.json
                                                          │          │  Event Log   │──▶ Event ID 1001-2001
                                                          ▼          └──────────────┘
                                                   ┌──────────────┐
                                                   │   Watchdog   │──▶ fails closed if the agent dies
                                                   └──────────────┘
```

The main loop captures frames from any webcam, detects faces (Haar cascades by default, YuNet DNN optional), and optionally recognizes known users via LBPH. Each frame is reduced to an `Observation` and fed into `PresenceStateMachine` (`lock_on_absence/state_machine.py`) — a pure-Python module with **zero dependencies** — which returns a `Verdict`:

- **KEEP** — presence proven, screen stays unlocked
- **WARN** — transition state, no action yet
- **LOCK** — lock the workstation for a specific `Reason` (intruder, absence, body-timeout, camera failure, spoof)
- **PAUSE** — camera is busy (video call); stop deciding, and resume the instant the camera is usable again

The agent loop is a dumb adapter: build `Observation` → `step()` → execute `Verdict`. Every decision, timer and threshold lives in the state machine, so the executed code is the tested code. An external watchdog (`lock-on-absence-watchdog`) locks the screen if the agent stops proving it is alive (stale heartbeat, dead PID, or clock tampering).

\* SFace (ONNX) recognition is on the roadmap — see [Roadmap](#roadmap).

---

## Installation

Requires **Python 3.10+**.

### Windows

```powershell
# 1. Install Python 3.10+ (https://python.org)
# 2. Open PowerShell (no admin needed)
cd C:\Users\handn\LockCam
.\install.ps1          # venv + Scheduled Tasks (agent + watchdog), opt-in
.\lock-on-absence-enroll  # capture your face
.\lock-on-absence
```

`install.ps1` registers two visible Scheduled Tasks (`LockOnAbsence`, `LockOnAbsenceWatchdog`) that start at logon. It deliberately does **not** drop a hidden script into the Startup folder — autostart you can see and stop.

### Linux

```bash
git clone https://github.com/handnewb/lock-on-absence.git
cd lock-on-absence
bash install.sh           # venv + systemd user units (agent + watchdog), opt-in
lock-on-absence-enroll    # capture your face
lock-on-absence
```

### macOS

```bash
git clone https://github.com/handnewb/lock-on-absence.git
cd lock-on-absence
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
lock-on-absence-enroll    # capture your face
lock-on-absence
```

---

## Quick start

```bash
# 1. Enroll your face (creates face_model.yml + face_model.json, chmod 600)
lock-on-absence-enroll

# 2. Run — fail-closed security mode (default)
lock-on-absence

# 3. Dry-run — decide and log, never lock (FAR/FRR measurement)
lock-on-absence --no-lock --siem lock-events.json

# 4. Measure before you trust a threshold
lock-on-absence-replay --synthetic --repeat 8 --far-window 12
```

All four commands also run from the repo root as `python lock-on-absence.py`, `python enroll.py`, `python watchdog.py`, `python replay.py`, or via `python -m lock_on_absence`.

---

## CLI reference

### `lock-on-absence` (agent)

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {security,convenience}` | `security` | `security` = fail-closed defaults (lock on camera failure, body-only ceiling); `convenience` = tolerant (warn instead of lock where safe) |
| `--delay` | `10` | Seconds with nobody present before locking |
| `--check-interval` | `1.5` | Seconds between camera reads |
| `--cooldown` | `30` | Quiet period after a lock (keep-awake never active during it) |
| `--max-body-only` | `20` | Max seconds with body-only detection before timeout lock |
| `--max-without-face` | `90` | Absolute ceiling since the last *recognized* face (immune to body-window refresh) |
| `--intruder-count` | `2` | Non-owner detections (sliding window) needed to lock |
| `--intruder-window` | `6` | Seconds for the sliding intruder window |
| `--startup-grace` | `5` | Suppress intruder lock right after start (never suppresses absence/camera) |
| `--on-camera-failure {lock,warn}` | `lock` | Fail-closed vs warn when the camera dies (mode overrides in `security`) |
| `--camera-fail-grace` | `20` | Seconds of camera failure before fail-closed lock |
| `--meeting-pause` | `30` | Maximum pause when another app holds the camera. Ends early the moment the camera is readable again, so a one-second grab does not buy 30s of blindness. Triggers after 5 busy checks; 15 min cumulative budget of real elapsed time. |
| `--anti-spoof-timeout` | `0` | Seconds of perfectly static face before a spoof lock (0 = disabled; weak heuristic, not liveness) |
| `--any-face` | — | Accept ANY face as owner. INSECURE: disables recognition-based intrusion detection |
| `--camera` | `0` | Camera index |
| `--model` | `./face_model.yml` | LBPH model path |
| `--yunet` | — | YuNet DNN detector instead of Haar cascades |
| `--stealth` | — | Open/close camera per frame. NOT RECOMMENDED: fails more, drains the same, LED still blinks |
| `--no-keep-awake` | — | Never suppress the OS idle timeout |
| `--no-lock` | — | Dry run: decide and log, never lock (events tagged `dry_run:true`) |
| `--event-log` | — | Write to Windows Event Log / syslog |
| `--siem` | — | Append JSON-lines events to this file |
| `--log-file` | — | Append log to this file |
| `--debug` | — | Log detection detail every ~28s |

### `lock-on-absence-enroll`

| Flag | Default | Description |
|------|---------|-------------|
| `--samples` | `30` | Face samples per user |
| `--camera` | `0` | Camera index |
| `--output` | `face_model.yml` | Output model path (metadata goes next to it) |
| `--users` | — | Comma-separated user names (e.g. `Alice,Bob`) |
| `--purge` | — | Delete model + metadata, revoke all enrolled users |
| `--no-consent` | — | Skip the consent prompt (automated enrollment) |

### `lock-on-absence-watchdog`

| Flag | Default | Description |
|------|---------|-------------|
| `--heartbeat` | `watchdog_heartbeat.txt` | Heartbeat file the agent writes every iteration |
| `--pid-file` | — | Also require this PID to be alive (LIVENESS check) |
| `--max-age` | `120` | Heartbeat older than this is stale (seconds) |
| `--interval` | `30` | Seconds between checks |
| `--once` | — | Check once and exit (cron / Scheduled Task) |
| `--dry-run` | — | Report what would happen; never lock |
| `--lock-missing` | — | Treat a *missing* heartbeat as stale (off by default: first boot has no file) |
| `--print-unit` / `--print-task` | — | Print an installable systemd unit / PowerShell Scheduled Task |

Design: **LATCH** (lock once per staleness episode, not every 30s forever), **SKEW-SAFE** (a future timestamp is tampering, not freshness), **LIVENESS** (prefers asking the OS about the process over trusting a user-writable file).

### `lock-on-absence-replay` (FAR/FRR harness)

| Flag | Description |
|------|-------------|
| `--synthetic` | Built-in 300s scenario, no files needed (CI-runnable) |
| `--video FILE` | Replay a real recording through the full vision pipeline |
| `--scenario FILE` | Replay a `.jsonl` of Observations (pure logic, deterministic) |
| `--record FILE` | With `--video`: write a scenario file instead of scoring |
| `--labels FILE` | Ground-truth CSV: `start_sec,end_sec,truth` (`owner`/`absent`/`intruder`/`body_only`) |
| `--repeat N` | Repeat the synthetic block for finer resolution |
| `--flicker F` | Drop this fraction of detections (simulate Haar losing faces at an angle) |
| `--sweep k=v,...` | Re-score across values, e.g. `delay=5,10,20` |
| `--fail-if-far-above X` | Exit 1 if FAR > X (CI gate) |
| `--fail-if-frr-above X` | Exit 1 if FRR > X (CI gate) |
| `--json FILE` | Write the report as JSON |

---

## FAR/FRR harness

Before v5.0, the recognition threshold was tuned by feel: `85 → 60 → 30 → 55 → 65` in a single day, none of it measured. The repo rule now: **nothing gets tuned again without a before/after table.**

Explicit metric definitions (each project uses these terms differently):

| Metric | Definition used here |
|---|---|
| **FAR** | fraction of `intruder` intervals in which the screen did **not** lock within `--far-window` |
| **FRR** | fraction of `owner` intervals in which the screen locked |
| **TTL** | seconds from the start of an `absent` interval to the lock (median and p90) |
| **spurious/h** | locks during `owner` per hour of owner presence |

Current numbers (default config, synthetic scenario, `--repeat 8`):

```
FAR  intruder missed   0.0%   (0/8 intervals)
FRR  owner rejected    0.0%   (0/32 intervals)
spurious locks/hour   0.00
time-to-lock median  11.0s
time-to-lock p90     11.5s
```

The harness also exposed the real bottleneck under bad detection: with 60% of detections dropped, raising `intruder_count` from 1 to 2 makes FAR jump to 25% — and varying the intruder *window* from 1.6s to 12s changes nothing. That is the kind of conclusion you cannot get from reading code.

**These numbers are NOT product validation.** The synthetic scenario measures the state machine, not the computer vision. The number that matters comes from *your* labeled video, in *your* lighting:

```bash
# 1. record ~30-60 min of your desk, then label it (owner/absent/intruder/body_only)
lock-on-absence-replay --video mesa.mp4 --record mesa.jsonl
lock-on-absence-replay --scenario mesa.jsonl --labels mesa.csv
lock-on-absence-replay --video mesa.mp4 --labels mesa.csv --model face_model.yml
```

The gates run in CI: any change that makes the machine miss an intruder or lock on the owner's face breaks the build. There is a test that verifies the gate *can* fail — a gate that cannot fail is not a gate.

---

## SIEM / event log

With `--siem <path>`, each lock event is appended as newline-delimited JSON:

```json
{"timestamp":"2026-07-30T10:15:30","event_id":1001,"event":"intruder_lock","message":"Screen locked: intruder detected (non-owner face)","hostname":"WORKSTATION-01","dry_run":false}
```

### Event ID reference

| ID | Name | Severity | Trigger |
|----|------|----------|---------|
| 1001 | `intruder_lock` | WARN | Non-owner face detected above the sliding-window threshold |
| 1002 | `absence_lock` | INFO | No face, no body — absence delay expired |
| 1003 | `spoof_lock` | WARN | Face static for `--anti-spoof-timeout` seconds |
| 1004 | `body_timeout_lock` | WARN | Body-only detection exceeded `--max-body-only` (or the 90s ceiling) |
| 1005 | `lock_failed` | ERROR | All lock mechanisms returned failure (emitted **instead of** the cause event, never alongside it) |
| 2001 | `camera_error` | ERROR | Persistent camera read failure → fail-closed lock |

All events carry `hostname` and `dry_run` (true when `--no-lock` is active), for cross-endpoint correlation in Splunk, Elastic, or Sentinel. With `--event-log`, the same events go to Windows Event Log (source: `LockOnAbsence`) or syslog (tag: `lock-on-absence`).

---

## Decision engine

`lock_on_absence/state_machine.py` is the **single source of truth** for every lock decision. It imports nothing from OpenCV, makes no OS calls, and every timer reads `Observation.t` — which is what makes the 28 state-machine tests run in milliseconds with no camera.

```python
from lock_on_absence.state_machine import (
    PresenceStateMachine, Config, Observation, State, Decision, Reason,
)

psm = PresenceStateMachine(Config(absence_delay=10.0))
st = State()

obs = Observation(t=time.monotonic(), faces=1, owner_recognized=True,
                  scene_unchanged=True, camera_ok=True)
verdict = psm.step(obs, st)
# verdict.decision == Decision.KEEP, verdict.reason == Reason.NONE
# verdict.message is populated only on a phase TRANSITION (no log spam)
# verdict.keep_awake tells the adapter whether to suppress the OS idle timeout
```

### Decision priority

1. **PAUSE** — camera busy; budget-capped at 15 min so it cannot hide a dead camera forever
2. **Cooldown** — recent lock blocks all checks; `keep_awake=False` even on KEEP
3. **Camera failure** — after `--camera-fail-grace` (20s default) → fail-closed **lock**
4. **Owner recognized** → **KEEP** (resets all timers)
5. **Intruder** → **LOCK** after N non-owner detections inside a *sliding window* (flicker-proof: a frame with no face does NOT clear the window)
6. **Absence** → **LOCK** after `--delay` without face or body
7. **Body-only** → **LOCK** after `--max-body-only`, hard-capped by `--max-without-face` since the last recognized face (the window cannot renew forever)

`Config.__post_init__` validates and enforces mode: `security` forces fail-closed camera behavior and clamps `max_body_only`; `convenience` downgrades lock to warn where safe. `Observation.__post_init__` rejects incoherent input (e.g. `owner_recognized` with zero faces) — an adapter bug becomes an exception, not a wrong silent decision.

### Tests (81 passing)

```bash
pip install -e ".[dev]"
python -m pytest -v
```

| Suite | Count | Validates |
|-------|-------|-----------|
| `tests/test_state_machine.py` | 28 | Every decision path, determinism, reason reachability, config validation, anti-spoof, pause budget, cooldown invariant |
| `tests/test_replay.py` | 43 | Harness scoring, gates (including that they *can* fail), watchdog (latch/skew/liveness), smoke tests (imports, instantiation, shims, `--help`), **architectural guard** |

The architectural guard (`test_agent_contains_no_presence_logic`) reads `agent.py` and fails if legacy decision variables — or any direct mutation of machine state (`st.x =`) — creep back into the adapter. It also verifies the root shims still work, so a deleted shim breaks CI.

---

## Architecture

```
lock-on-absence/
├── pyproject.toml              # package definition, 4 entry points, deps, tooling
├── lock_on_absence/
│   ├── __init__.py             # __version__ = "5.0.0" (single source)
│   ├── __main__.py             # python -m lock_on_absence
│   ├── agent.py                # dumb adapter: frame → Observation → Verdict → execute
│   ├── state_machine.py        # pure decision logic (no deps, fully tested)
│   ├── face_utils.py           # detectors, LBPH recognizer, KeepAwake, lock_screen, SIEM
│   ├── enroll.py               # face enrollment, --purge, --no-consent
│   ├── replay.py               # FAR/FRR harness (synthetic / video / record / sweep)
│   └── watchdog.py             # external watchdog (LATCH + SKEW-SAFE + LIVENESS)
├── lock-on-absence.py          # root shims (8 lines) — keep old commands working
├── enroll.py                   #   (installers, README, muscle memory)
├── watchdog.py
├── replay.py
├── tests/
│   ├── test_state_machine.py   # 28 tests
│   └── test_replay.py          # 43 tests
├── install.sh                  # Linux: venv + systemd user units (opt-in)
├── install.ps1 / install.bat   # Windows: visible Scheduled Tasks (opt-in)
├── .github/workflows/ci.yml    # matrix 3 OS × 2 Pythons + FAR/FRR gates
├── MIGRATION.md                # v4.1 → v5.0 migration record
└── LICENSE                     # MIT
```

### Key design decisions

1. **The state machine is the product** — every decision is a pure function of `(Observation, State)`, which is why 81 tests run in ~7s with no camera and the executed code is the tested code.
2. **The agent owns no timers and no branches** — any new `if` about presence belongs in `state_machine.py`, not `agent.py` (enforced by CI).
3. **Fail-closed by default** — dead camera, crashed agent (watchdog), or unhandled exception in the loop all end in a locked screen, never an open one.
4. **Nothing is tuned without a measurement** — the FAR/FRR harness replaced guess-based threshold tuning; CI gates block regressions.
5. **SIEM is the evidence trail** — who was at the workstation, when they left, and whether the screen locked, in structured JSON with `dry_run` tagging.
6. **Honest security claims** — no blink-level liveness; the movement anti-spoof is documented as a weak heuristic and off by default.
7. **All internal durations use `time.monotonic()`** — immune to system clock jumps. The heartbeat file uses `time.time()` only for watchdog compatibility, with future-timestamp tampering treated as suspicious.
8. **Installers are visible** — opt-in autostart via systemd units / Scheduled Tasks, never a hidden startup script (EDR/stalkerware signature).

---

## What v5.1 changed

Three findings from the post-v5.0 audit, all with regression tests:

**Pause no longer means blind.** The adapter reads the camera every tick, so
`Observation.camera_ok` already told the state machine when the device came back —
and the pause branch returned before looking at it. Any process that grabbed the
webcam for one second bought a full `--meeting-pause` of blindness, during which
an intruder could not be locked out either. Measured before the fix: 26 seconds of
`PAUSE` with the camera free and the owner's face visible. After: 0. The pause
budget now also counts real elapsed time instead of the nominal window, so
`meeting_pause_max` means real seconds.

**Mode clamps are no longer silent.** `--mode security` overrides settings that
would weaken it — `--max-body-only 60` becomes 20, `--on-camera-failure warn`
becomes `lock`. It used to do that without a word, so you believed a flag took
effect when it had not. Every override is now recorded in `Config.clamps` and
logged at startup as `NOTE: max_body_only=60 clamped to 20 by mode=security`.

**The model file is integrity-checked.** `enroll` records a SHA-256 of
`face_model.yml` in `face_model.json`, and the agent refuses to start (exit 3) if
they disagree. Be precise about what this buys: it is tamper **detection**, not
prevention. An attacker who can rewrite the model can usually rewrite the JSON
too. What it does buy is catching corruption, and forcing any tamper to be
consistent across two files instead of one. Real prevention needs the key in an OS
keyring (DPAPI / Keychain / Secret Service) — that is on the roadmap, not done.

File permissions are also fixed on Windows: `os.chmod(path, 0o600)` is close to a
no-op on NTFS, where inherited ACLs can leave the file readable by other local
accounts. `restrict_file_permissions()` now uses `icacls /inheritance:r` there,
and reports honestly when it cannot.

---

## Security model

| Property | Current | Target |
|----------|---------|--------|
| Fail-closed on camera failure | ✅ `--on-camera-failure lock` (20s grace) | — |
| Fail-closed on crash | ✅ Watchdog (LATCH + skew-safe) + crash handler locks before exit | systemd `WatchdogSec` + `sd_notify` (already wired) |
| FAR/FRR measurement | ✅ `lock-on-absence-replay` + CI gates | Published benchmark from real labeled video |
| Face template protection | ✅ `chmod 600` after enrollment | `icacls` on Windows, HMAC on `face_model.yml` |
| Threshold integrity | ✅ `face_model.json` clamped to [20, 100] (user-writable file cannot force 999999) | — |
| Video-call coexistence | ✅ Camera-busy pause (5 checks → pause, 15 min budget) | — |
| Liveness detection | ⚠️ Movement-based anti-spoof, off by default | YuNet 5-point landmarks → real yaw |
| External watchdog | ✅ heartbeat + PID check | — |
| Multi-factor | ❌ Webcam only | TESSERA/BLE token |

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | clean shutdown (Ctrl+C / SIGTERM) |
| `1` | camera could not be opened, or the loop crashed (screen is locked before exit) |
| `2` | no face model and `--any-face` not given — refusing to run unprotected |
| `3` | `face_model.yml` does not match the digest recorded at enrollment |

---

## Limitations

- **Facial recognition uses LBPH (2006)** — works in consistent indoor lighting with frontal faces; not competitive with modern embeddings. SFace ONNX (`cv2.FaceRecognizerSF`, published cosine threshold ~0.363) is the planned replacement.
- **`estimate_angle` measures position in frame, not pose** — it detects "face left the center", not "head turned". Real yaw needs YuNet landmarks.
- **Threshold 65 is inherited, not measured** — the current LBPH config (`radius=2`, grid `6x6`) changed the chi-square distance scale. It stays until a real labeled dataset produces a before/after table.
- **One webcam at a time** — the camera-busy pause handles video calls, but shared peripherals and hot-swapping are not handled.
- **Biometric data is stored in plain files** — `face_model.yml` (LBPH histograms) and `face_model.json` (user names) are unencrypted, protected only by `chmod 600`. HMAC integrity is on the roadmap. Use at your own risk in regulated environments.
- **Privacy consent** — `enroll.py` prompts for consent before capture; `--no-consent` exists for automated enrollment.

---

## Roadmap

- [ ] **Real labeled dataset** — 30–60 min of labeled desk video per lighting condition; publish the FAR/FRR table in this README. The harness is a chassis without an engine until then.
- [ ] **YuNet + SFace recognition** — replaces Haar+LBPH; kills the entire threshold-bug category; YuNet landmarks give real yaw.
- [ ] **HMAC on `face_model.yml`** — the model file is the product's trust boundary; today it is a plain file.
- [ ] **`icacls` for Windows model permissions** — `chmod 600` is POSIX-only.
- [ ] **CEF / LEEF output** for Splunk native ingestion.

See [open issues](https://github.com/handnewb/lock-on-absence/issues) for the full list.

---

## Contributing

PRs are welcome. Follow the existing style (typed Python, KISS, DRY) and add tests to `tests/test_state_machine.py` for any new decision path — the architectural guard and the FAR/FRR gates will check the rest.

Areas that need contribution:
- **Dataset** — labeled video captures (owner present, intruder, empty chair, backlight, oblique angles)
- **SFace integration** — swap LBPH for `cv2.FaceRecognizerSF` with the published cosine threshold
- **Liveness** — real pose estimation from YuNet 5-point landmarks
- **Model integrity** — HMAC signing of `face_model.yml`
- **SIEM formats** — CEF/LEEF output

---

## License

MIT © 2026 [Everton (handnewb)](https://github.com/handnewb)

See [LICENSE](LICENSE) for the full text.
