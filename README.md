# Lock on Absence

> Auto-lock your screen when you walk away. Optional facial recognition so only
> **you** can keep it unlocked. Works with any webcam — no special hardware.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Quick Start](#quick-start)
- [Detection Layers](#detection-layers)
- [Installation](#installation)
- [Enrolling Your Face](#enrolling-your-face)
- [CLI Reference](#cli-reference)
- [Auto-Start](#auto-start)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)
- [License](#license)

---

## What It Does

Point your webcam at where you sit. The script monitors the feed and decides
whether to lock the screen based on **four independent detection layers**:

| Layer | Trigger | Action |
|-------|---------|--------|
| **Owner face** | Your face detected & recognized | Screen stays **unlocked** — system keeps awake |
| **Intruder** | A face that is NOT you (2 consecutive frames) | Screen **locks immediately** — zero delay |
| **Body present** | No face visible, but the scene hasn't changed | Screen stays **unlocked** — assumes you turned away |
| **Nobody home** | No face AND scene changed significantly | Screen **locks** after configurable delay |

This means:

- You can look **sideways, down, or even turn your back** — the system sees that
  your body is still in the chair and won't lock.
- If someone else sits down, the screen locks **instantly** — before they can
  touch anything.
- If you get up and leave, the room scene changes and the countdown begins.
- While you're present, the system prevents Windows/Linux from sleeping or
  locking — no more wiggling the mouse to keep the screen alive.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Recommended) Enroll your face — 40 samples, ~20 seconds
python enroll.py --samples 40

# 3. Start monitoring
python lock-on-absence.py
```

That's it. Walk away and the screen locks after 10 seconds.

---

## Detection Layers

### Layer 1 — Owner Face Recognition

When your face is detected AND recognized as the enrolled owner, all lock timers
are reset. The system also keeps the display awake (see [Keep-Awake](#keep-awake)).

Uses **LBPH** (Local Binary Patterns Histogram) trained on your face samples.
Multi-angle cascades detect frontal, left profile, and right profile faces.

### Layer 2 — Intruder Detection

If a face is detected but the LBPH model does **not** recognize it as the owner,
the system requires **2 consecutive frames** (~3 seconds) of non-recognition to
confirm an intruder. Once confirmed: **instant lock**. No delay.

This 2-frame window prevents the owner from being locked out when the model
momentarily fails at an unusual head angle.

### Layer 3 — Body Presence Detection

What happens when you look down at your phone, turn completely sideways, or
lean back so far your face is out of frame? The system compares the current
camera frame against a reference frame captured when you were last recognized.

If the scene hasn't changed significantly (mean absolute difference < 18 at
160×120 resolution), the system assumes you're still there and keeps the screen
unlocked. Small movements, posture shifts, and lighting changes are tolerated.

The reference frame is refreshed every 30 seconds while your face is visible.

### Layer 4 — Absence Timer

When no face is detected AND the scene has changed (you actually left), a
countdown begins. After the configured delay (default: 10 seconds), the screen
locks. This is the fallback for genuine absence.

---

## Installation

### Prerequisites

- Python 3.8 or later
- A working webcam (built-in or USB)
- Windows, Linux, or macOS

### One-Command Setup

**Windows:**
```cmd
install.bat
```
Installs dependencies, tests your webcam, and adds a startup entry so the
script launches automatically on login.

**Linux / macOS:**
```bash
chmod +x install.sh
./install.sh
```
Creates a `systemd --user` service for auto-start on login.

### Manual Setup

```bash
pip install -r requirements.txt
# Optional: enroll your face
python enroll.py --samples 40
# Start
python lock-on-absence.py
```

### Dependencies

```
opencv-python>=4.8,<5       # face detection (Haar cascades)
opencv-contrib-python>=4.8,<5  # face recognition (LBPH)
numpy>=1.24                 # array operations
```

---

## Enrolling Your Face

Enrollment captures multiple face samples and trains a model that only
recognizes **you**.

```bash
python enroll.py                   # 30 samples (default, ~15s)
python enroll.py --samples 50      # 50 samples (more angles, ~25s)
python enroll.py --camera 1        # different camera
python enroll.py --output ~/my_face.yml  # custom output path
```

### During enrollment

Move your head through ALL angles — slowly:

```
FRONT → LEFT profile (90°) → FRONT → RIGHT profile (90°) → UP → DOWN → repeat
```

The progress bar shows which angle is being detected:

```
[##############################] 40/40 (FRONT)
```

Tips for a good model:

- **Lighting**: even frontal light, avoid strong backlight
- **Background**: plain wall preferred, avoid photos/posters with faces
- **Glasses**: remove if you switch between glasses/no-glasses during the day
- **Distance**: sit at your normal working distance (~60cm–1m from camera)

The trained model is saved as `face_model.yml` in the script directory.

---

## CLI Reference

### `lock-on-absence.py`

```
python lock-on-absence.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--delay SECONDS` | `10` | Seconds without owner before locking |
| `--camera INDEX` | `0` | Webcam index (try `1` if built-in doesn't work) |
| `--no-lock` | off | Dry-run mode — detect & log, but never lock |
| `--no-keep-awake` | off | Disable keep-awake (allow normal sleep/lock) |
| `--check-interval SECONDS` | `1.5` | Time between detection frames |
| `--model PATH` | `./face_model.yml` | Path to trained LBPH model |
| `--cooldown SECONDS` | `30` | Silence period after locking before monitoring resumes |
| `--log-file PATH` | off | Also write timestamped log to a file |
| `--max-body-only SECONDS` | `60` | Max time body detection holds unlock without face re-verification |

### `enroll.py`

```
python enroll.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--samples N` | `30` | Number of face samples to capture |
| `--camera INDEX` | `0` | Webcam index |
| `--output PATH` | `./face_model.yml` | Where to save the model |

### Running in the Background

**Windows** (no terminal window):
```cmd
pythonw lock-on-absence.py --delay 7
```

**Linux** (via systemd):
```bash
systemctl --user start lock-on-absence
```

**Stop:**
```bash
# Windows
taskkill /f /im pythonw.exe

# Linux
systemctl --user stop lock-on-absence
```

---

## Auto-Start

### Windows

Run `install.bat`. It creates a VBS wrapper in the Startup folder:

```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\LockOnAbsence.vbs
```

The script launches silently on every login.

### Linux

Run `install.sh`. It creates a user-level systemd service:

```
~/.config/systemd/user/lock-on-absence.service
```

Enabled automatically. Manage with:

```bash
systemctl --user start lock-on-absence
systemctl --user stop lock-on-absence
systemctl --user disable lock-on-absence   # remove from auto-start
```

---

## How It Works

### Face Detection

Uses OpenCV's **Haar Cascade** classifiers. Two cascades are loaded:

- `haarcascade_frontalface_default.xml` — front-facing faces
- `haarcascade_profileface.xml` — left profile faces

Right profiles are detected by **mirroring** the frame and running the same
profile cascade, then un-flipping the coordinates.

Parameters tuned for range and reliability:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `scaleFactor` | 1.03 | Fine pyramid scan (finds small/distant faces) |
| `minNeighbors` | 2 | High sensitivity (safe with streak confirmation) |
| `minSize` | 30×30 px | Detects faces up to ~1.5 meters away |
| Resolution | 640×480 | Native webcam resolution (validated 100% detection) |

### Face Recognition

Uses OpenCV's **LBPHFaceRecognizer** (Local Binary Patterns Histogram). LBPH is
lightweight, fast, and works well with small training sets (30–50 images).

- Recognition threshold: confidence < 85 = owner match
- Model training: `enroll.py` captures samples → trains → saves `face_model.yml`

### Body Presence Detection

Uses frame differencing (`cv2.absdiff`) between the current frame and a
reference frame captured when the owner was last recognized.

- Frames are downscaled to 160×120 for speed and noise reduction
- Mean absolute difference < 18 → scene is unchanged → body still present
- Reference updated every 30 seconds (adapts to gradual lighting changes)

### Keep-Awake

Prevents the operating system from sleeping or locking the display while the
owner is present:

| Platform | Mechanism |
|----------|-----------|
| Windows | `SetThreadExecutionState(ES_DISPLAY_REQUIRED \| ES_SYSTEM_REQUIRED \| ES_CONTINUOUS)` |
| Linux | `systemd-inhibit --what=idle:sleep` |
| macOS | `xdg-screensaver reset` (fallback) |

When the owner leaves, the keep-awake is released and the system resumes its
normal power-saving behavior.

### Screen Lock

| Platform | Mechanism |
|----------|-----------|
| Windows | `LockWorkStation()` |
| Linux | `loginctl lock-session` → `xdg-screensaver lock` → `i3lock` → `slock` |
| macOS | `osascript -e 'tell application "System Events" to sleep'` |

---

## Troubleshooting

### "Camera not available"

- Check that no other app is using the webcam (Zoom, Teams, browser)
- Try a different index: `--camera 1`
- Windows: make sure camera permissions are enabled in Settings > Privacy

### "It locks even when I'm sitting there"

- Lighting may have changed dramatically (sunlight, lamp turned off)
- Try re-enrolling with current lighting: `python enroll.py --samples 40`
- Increase delay: `--delay 15`
- Body detection needs ~30s of face recognition before it has a reference frame

### "It doesn't lock when I leave"

- Something in the background may be detected as a face (poster, photo, mirror)
- Body detection may see the empty chair as "unchanged" — try `--delay 5`
- Check the terminal output for "Body still present" messages

### "Multiple faces detected" during enrollment

- Remove photos, posters, or reflective surfaces from the camera's view
- Ensure you're the only person in the frame
- The warning is informational — single-face samples are still captured

### "Module not found: cv2"

```bash
pip install -r requirements.txt
# If that fails:
pip install opencv-python==4.14.0.94 opencv-contrib-python==4.14.0.94
```

### High CPU usage

The script uses ~2–5% CPU on modern hardware. Resolution (640×480) and check
interval (1.5s) are already optimized. Reduce further with:

```bash
python lock-on-absence.py --check-interval 3
```

---

## Changelog

### v3.1 — Body-Only Timeout + Security Hardening (current)
- **New:** `--max-body-only` flag — max time body detection holds unlock without face re-verification (default 60s)
- **Security fix:** body reference frame only updates when face IS recognized (was: updated during body-only mode, allowing attacker's body to become the new reference)
- **Security fix:** mandatory periodic face re-verification — body detection alone cannot hold the screen unlocked indefinitely
- Body-detect log now shows countdown: "body present, re-verify in 45s"
- Lock reason now explicit: "body-only timeout: 65s > 60s"

### v3.0 — Refactor + Keep-Awake Fix + Auto-Calibration
- **New:** `face_utils.py` — shared module (face detection, body detection, camera, logging)
- **Fixed:** Linux keep-awake now uses a persistent `systemd-inhibit` subprocess (was broken — `true` exited immediately)
- **Fixed:** macOS keep-awake via `caffeinate` subprocess
- **New:** Body detection auto-calibration — adapts threshold to camera/lighting
- **New:** `--cooldown` flag (was hardcoded 30s)
- **New:** `--log-file` flag for persistent logging
- **New:** `KeepAwake` class, `BodyDetector` class, `Logger` class
- **New:** Signal handlers (SIGINT/SIGTERM) for graceful shutdown
- **New:** Camera backend fallback (V4L2 → default on Linux)
- **Improved:** `enroll.py` uses shared `face_utils` module
- **Reduced:** 342 lines of duplicated code eliminated

### v2.4 — Body Presence Detection
- Layer 3: frame differencing detects when the owner's body is still in the chair
- Reference frame captured when owner's face is recognized, refreshed every 30s
- Mean absdiff < 18 at 160×120 → body present → unlocked
- New CLI label: `[body-detect ON]`

### v2.3 — Right Profile + Long Range
- Right profile detection via frame mirroring
- minSize reduced from 40→30px (faces up to ~1.5m)
- scaleFactor tightened from 1.05→1.03 (finer scan)
- minNeighbors relaxed from 3→2 (more sensitive)
- Recognition threshold raised from 75→85 (more tolerant)

### v2.2 — Intruder Streak Confirmation
- Intruder now requires 2 consecutive non-owner frames before locking
- Prevents owner from being locked out at extreme head angles
- 1-frame recognition misses are tolerated

### v2.1 — Instant Intruder Lock
- Face detected but NOT owner → immediate lock (was: same delay as nobody)

### v2.0 — Multi-Angle Detection + Keep-Awake
- Added `haarcascade_profileface.xml` for side-profile detection
- Keep-awake mode (Windows: `SetThreadExecutionState`, Linux: `systemd-inhibit`)
- `--no-keep-awake` flag
- Enroll shows angle feedback (FRONT/LEFT/RIGHT)

### v1.0 — Initial Release
- Haar Cascade frontal face detection
- LBPH facial recognition (`enroll.py`)
- Cross-platform screen lock (Windows/Linux/macOS)
- One-click installers (`install.bat`, `install.sh`)
- Auto-start (Startup folder, systemd user service)

---

## License

MIT — use it, fork it, ship it.

---

## Contributing

Issues and pull requests welcome. Areas that need love:

- macOS testing (body detection, keep-awake)
- DNN-based face detection (YuNet) for better accuracy
- Anti-spoofing (prevent photo/video bypass)
- Multiple owner profiles
