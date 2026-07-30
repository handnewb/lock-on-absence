# Lock on Absence

> Auto-lock your workstation when you walk away. Facial recognition, SIEM audit trail, and a pure-Python decision engine — no special hardware required.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-Windows%20|%20Linux%20|%20macOS-lightgrey)](https://github.com/handnewb/lock-on-absence)
[![Version](https://img.shields.io/badge/version-4.1.0-brightgreen)](https://github.com/handnewb/lock-on-absence/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-7%20passed%20(v4.1)-success)](test_state_machine.py)

---

## Table of Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
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
┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐
│  Webcam  │──▶│  Detector    │──▶│  Recognizer │──▶│  State   │
│ (Haar /  │   │ (Haar /      │   │ (LBPH /     │   │ Machine  │──▶ Lock / Keep
│  YuNet)  │   │  YuNet)      │   │  SFace*)    │   │          │
└──────────┘   └──────────────┘   └────────────┘   └──────────┘
                                                      │
                                                      ▼
                                                ┌──────────────┐
                                                │  SIEM Output │──▶ lock-events.json
                                                │  Event Log   │──▶ Event ID 1001-1005
                                                └──────────────┘
```

The main loop captures frames from any webcam, detects faces (Haar cascades by default, YuNet DNN optional), and optionally recognizes known users via LBPH. Each frame is reduced to an `Observation` and fed into a pure Python state machine (`presence_state_machine.py`) that decides whether to:

- **KEEP** — screen unlocked, presence proven
- **WARN** — transition state, no action yet
- **LOCK** — lock the workstation for a specific `Reason` (intruder, absence, body-timeout, camera failure)

All decisions are logged and optionally written as structured JSON to stdout or a SIEM file.

\* SFace (ONNX) recognition is on the roadmap — see [§4.4 of the adversarial review](https://github.com/handnewb/lock-on-absence/issues/1).

---

## Installation

### Windows

````powershell
# 1. Install Python 3.9+ (https://python.org)
# 2. Open PowerShell as Administrator
cd C:\Users\handn\LockCam
pip install -r requirements.txt
python enroll.py  # capture your face
python lock-on-absence.py
````

### Linux

````bash
git clone https://github.com/handnewb/lock-on-absence.git
cd lock-on-absence
bash install.sh           # venv + systemd service (interactive)
python3 enroll.py         # capture your face
./.venv/bin/python lock-on-absence.py
````

### macOS

````bash
git clone https://github.com/handnewb/lock-on-absence.git
cd lock-on-absence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python enroll.py
python lock-on-absence.py
````

---

## Quick start

````bash
# 1. Enroll your face (creates face_model.yml + face_model.json)
python enroll.py

# 2. Run (lock on absence, default settings)
python lock-on-absence.py

# 3. Dry-run mode (no actual locks — log only)
python lock-on-absence.py --no-lock

# 4. Run with SIEM output
python lock-on-absence.py --siem lock-events.json
````

---

## CLI reference

### `lock-on-absence.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Camera index to use |
| `--model` | `face_model.yml` | Path to trained LBPH model |
| `--threshold` | `65` | Recognition confidence threshold (lower = stricter) |
| `--absence-delay` | `10` | Seconds without owner before lock |
| `--cooldown` | `30` | Seconds after a lock before re-checking |
| `--max-body-only` | `20` | Max seconds with body-only detection before timeout lock |
| `--anti-spoof-timeout` | `0` | Seconds of static face before anti-spoof trigger (0 = disabled) |
| `--on-camera-failure` | `lock` | Behavior on extended camera failure (`lock`, `warn`, `ignore`) |
| `--siem` | — | Path to structured JSON event log (appended) |
| `--event-log` | — | Enable OS event logging (Windows Event Log / syslog) |
| `--no-lock` | — | Dry-run: log decisions without locking (for FAR/FRR measurement) |
| `--any-face` | — | Allow any detected face to keep screen unlocked (no model needed) |
| `--meeting-pause` | `30` | Seconds to pause when camera is in use by another app |
| `--yunet` | — | Use YuNet DNN detector instead of Haar cascades |
| `--stealth` | — | Open/close camera per frame (LED blinks instead of stays on) |
| `--debug` | — | Verbose logging every ~30s |
| `--log-file` | — | File path for log output |

### `enroll.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `face_model.yml` | Output model filename |
| `--samples` | `30` | Number of face captures to train on |
| `--camera` | `0` | Camera index |
| `--users` | — | Comma-separated user names (e.g. `Alice,Bob`) |
| `--purge` | — | Delete face model and metadata, revoke all enrolled users |
| `--no-consent` | — | Skip privacy consent prompt (automated enrollment) |

### Multi‑user enrollment

To enroll multiple users (e.g., family members or co‑workers sharing the desk):

```bash
# Enroll Alice and Bob in one pass
python enroll.py --users Alice,Bob --samples 50

# Run with Alice's model
python lock-on-absence.py --model face_model_Alice.json

# Run with Bob's model
python lock-on-absence.py --model face_model_Bob.json
```

Each user gets a separate `face_model_<name>.yml` + `face_model_<name>.json` pair.
Pass `--model` to select which face unlocks the screen.  
Users not selected by `--model` are **ignored** — they count as non-owner faces (intruder).

---

## SIEM / event log

When `--siem <path>` is passed, each lock event is appended as a newline-delimited JSON record:

```json
{"timestamp":"2026-07-30T10:15:30","event_id":1001,"event":"intruder_lock","message":"Screen locked: intruder detected (non-owner face)","hostname":"WORKSTATION-01"}
```

### Event ID reference

| ID | Name | Severity | Trigger |
|----|------|----------|---------|
| 1001 | `intruder_lock` | WARN | Non-owner face detected above streak threshold |
| 1002 | `absence_lock` | INFO | No face, no body — absence delay expired |
| 1003 | `spoof_lock` | WARN | Face static for `--anti-spoof-timeout` seconds |
| 1004 | `body_timeout_lock` | WARN | Body-only detection exceeded `--max-body-only` |
| 1005 | `lock_failed` | ERROR | All lock mechanisms returned failure |
| 2001 | `camera_error` | ERROR | Persistent camera read failure |

All events include `hostname`, used for cross-endpoint correlation in SIEM ingestion (Splunk, Elastic, Sentinel). When `--event-log` is also enabled, the same events go to Windows Event Log (source: `LockOnAbsence`) or syslog (tag: `lock-on-absence`).

---

## Decision engine

`presence_state_machine.py` is a pure-Python state machine extracted from the main loop per the [adversarial review §4.6](https://github.com/handnewb/lock-on-absence/issues/1). It has **zero dependencies** (no OpenCV, no OS calls) and is fully testable with `pytest`.

````python
from presence_state_machine import PresenceStateMachine, Config, Observation, State, Decision, Reason

sm = PresenceStateMachine(Config(absence_delay=10.0))
st = State()

obs = Observation(t=time.time(), faces=1, owner_recognized=True,
                  scene_unchanged=True, camera_ok=True)
decision, reason = sm.step(obs, st)
# decision == Decision.KEEP, reason == Reason.NONE
````

### Decision priority

1. **Cooldown** — recent lock blocks all checks
2. **Camera failure** — `--camera-fail-grace` = 20s → fail-closed **lock** (default)
3. **Owner recognized** → **KEEP** (reset all timers)
4. **Intruder streak** → **LOCK** after N consecutive non-owner frames (N=2 default)
5. **Absence** → **LOCK** after `absence_delay` without face or body
6. **Body-only** → **LOCK** after `max_body_only` with scene unchanged but no face

### Tests (7/7 passing)

````bash
python -m pytest test_state_machine.py -v
````

| Test | Validates |
|------|-----------|
| `test_owner_present_keeps` | Owner resets all timers, returns KEEP |
| `test_intruder_streak_locks` | N consecutive non-owner frames trigger LOCK |
| `test_absence_locks_after_delay` | Absence timer triggers LOCK |
| `test_body_only_timeout` | Body-without-face timeout triggers LOCK |
| `test_camera_failure_fail_closed` | Camera failure after grace → LOCK |
| `test_cooldown_blocks` | Recent lock prevents re-triggering |
| `test_recovery_resets` | Owner re-appearance clears intruder streak |

---

## Architecture

```
lock-on-absence/
├── lock-on-absence.py          # Main loop, CLI, state machine integration
├── face_utils.py               # Detectors (Haar, YuNet), recognizer (LBPH),
│                               #   KeepAwake, lock_screen, EventLogger, Logger
├── enroll.py                   # Face enrollment, training, --purge
├── presence_state_machine.py   # Pure decision logic (no deps, testable)
├── watchdog.py                 # External watchdog (heartbeat → stale detection)
├── test_state_machine.py       # pytest suite for state machine
├── install.sh                  # Linux systemd installer with venv
├── requirements.txt            # opencv-contrib-python, numpy
├── LICENSE                     # MIT
├── __init__.py                 # __version__ = "4.1.0"
└── .gitignore
```

### Key design decisions

1. **State machine extracted from main loop** — Decisions are deterministic functions of `Observation`. Without this, every camera test requires a physical webcam. With it, the 7 pytest cases run in 0.05s and cover all lock paths.
2. **Fail-closed by default** — `--on-camera-failure lock` means a dead camera locks the workstation, not leaves it open. See [C3 in the adversarial review](https://github.com/handnewb/lock-on-absence/issues/1).
3. **SIEM is the product** — The most defensible feature is the evidential trail: who was at the workstation, when did they leave, did the system lock. Structured JSON + OS event log + hostname enables cross-endpoint correlation.
4. **Watchdog as second layer** — Heartbeat file written every 60s, `watchdog.py` reads it and locks if stale >120s. Catches crashes, zombs, and kills that the main loop can't handle.
5. **No anti-spoof claims** — The eye-cascade blink detector was removed in v4.0 because it was structurally incapable of detecting a blink. Claims about photo detection have been removed. See [C5](https://github.com/handnewb/lock-on-absence/issues/1).

---

## Security model

| Property | Current | Target |
|----------|---------|--------|
| Fail-closed on camera failure | ✅ Default `lock` | — |
| Fail-closed on crash | ✅ Watchdog | — |
| SIEM audit trail | ✅ JSON + Event Log | CEF/LEEF |
| Face template protection | — `face_model.json` is world-readable | `chmod 600`, DPIA |
| FAR/FRR measurement | 🔧 `--no-lock` + SIEM | Published benchmark |
| Liveness detection | ❌ Removed in v4.0 | MediaPipe landmarks |
| External watchdog | ✅ `watchdog.py` | systemd `WatchdogSec` |
| Multi-factor | ❌ Webcam only | TESSERA/BLE token |

See [adversarial review — Issue #1](https://github.com/handnewb/lock-on-absence/issues/1) for the full security audit.

---

## Limitations

- **Facial recognition uses LBPH (2006)** — it works in consistent indoor lighting with frontal faces but is not competitive with modern embeddings. SFace ONNX is the planned replacement (§4.4).
- **False locks possible** — threshold tuning is environment-specific. Use `--no-lock` + `--siem` to measure FAR/FRR before deploying.
- **One webcam at a time** — the camera is a shared resource. `--meeting-pause` avoids conflicts with video calls, but switching users or sharing peripherals isn't handled.
- **macOS lock is screen blanking** — macOS doesn't expose a reliable `LockWorkStation()` equivalent without Accessibility permissions. The fallback keystroke + `pmset displaysleepnow` blanks the screen but does not require authentication to wake.
- **No CI/CD yet** — tests pass locally. GitHub Actions pipeline planned.
- **Biometric data is stored in plain files** — `face_model.yml` (LBPH histograms) and `face_model.json` (user names) are not encrypted. Use at your own risk in regulated environments.

---

## Roadmap

- [ ] **FAR/FRR harness** — replay recorded video with injected timestamps to measure error rates
- [ ] **SFace ONNX recognition** (`cv2.FaceRecognizerSF`) — transferable threshold, no LBPH
- [ ] **GitHub Actions CI** — pytest on push + PR
- [ ] **systemd `WatchdogSec=30`** integration with `sd_notify`
- [ ] **Windows Scheduled Task watchdog** — lightweight `watchdog.py` scheduled via PowerShell
- [ ] **`chmod 600` + encryption** for face templates
- [ ] **CEF / LEEF output** for Splunk native ingestion

See [open issues](https://github.com/handnewb/lock-on-absence/issues) for the full list.

---

## Contributing

PRs are welcome. Please follow the existing code style (typed Python, KISS, DRY) and add tests to `test_state_machine.py` for any new decision paths.

Areas that need contribution:
- **Dataset** — labeled video captures for FAR/FRR measurement (owner present, intruder, empty chair, backlight, oblique angles)
- **MediaPipe liveness** — real EAR or active-challenge blink detection
- **CI/CD** — GitHub Actions workflow for pytest + lint
- **SFace integration** — swap LBPH for `cv2.FaceRecognizerSF` with published cosine threshold

---

## License

MIT © 2026 [Everton (handnewb)](https://github.com/handnewb)

See [LICENSE](LICENSE) for the full text.
