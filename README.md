# Lock on Absence

Auto-lock your screen when you walk away from your computer — using just your webcam.

**Two modes:**

| Mode | Behavior |
|------|----------|
| **Any face** (default) | Locks when NO face is detected. Any person in front keeps it unlocked. |
| **Owner recognition** | Locks when YOUR face is NOT detected. A different person triggers the lock too. |

## Features

- **Multi-angle detection** — frontal + left/right profile (Haar cascades)
- **Facial recognition** — LBPH model, trainable via `enroll.py`
- **Keep-awake** — prevents sleep/lock while you're present (Windows + Linux)
- **Cross-platform** — Windows, Linux, macOS
- **One-click installers** — `install.bat` (Windows), `install.sh` (Linux)
- **Auto-start** — Startup folder (Windows), systemd user service (Linux)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Recommended) Enroll your face
python enroll.py

# 3. Start monitoring
python lock-on-absence.py
```

Walk away from the camera. After 10 seconds the screen locks.

## How It Works

1. Captures frames from your webcam every ~1.5s
2. Detects faces from multiple angles (frontal + profile cascades)
3. If owner recognition is enabled (model exists), checks whether the face matches
4. When the owner is gone for N seconds → locks the screen
5. While the owner is present → prevents system sleep/lock (keep-awake)

On **Windows**: calls `LockWorkStation()` + `SetThreadExecutionState` for keep-awake  
On **Linux**: calls `loginctl lock-session` + `systemd-inhibit` for keep-awake  
On **macOS**: puts display to sleep

## CLI Options

```
python lock-on-absence.py --help

  --delay 5            Seconds without owner before lock (default: 10)
  --camera 1           Use camera index 1
  --no-lock            Dry-run: detect but never lock
  --no-keep-awake      Disable keep-awake (allow normal sleep/lock)
  --check-interval 2   Seconds between checks (default: 1.5)
  --model face.yml     Path to trained model
```

## Enrolling Your Face

```bash
python enroll.py                   # 30 samples (~20s)
python enroll.py --samples 50      # 50 samples, more accurate
```

Move your head through all angles:

```
FRONT → LEFT profile → RIGHT profile → UP → DOWN
```

The progress bar shows which angle was detected (FRONT/LEFT/RIGHT).

The model is saved as `face_model.yml` in the script directory.

**Important:** after updating the scripts, re-enroll for best profile recognition:
```bash
python enroll.py --samples 50
```

## Keep-Awake Mode

Enabled by default. While the owner is detected:
- **Windows**: calls `SetThreadExecutionState(ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED)`
- **Linux**: uses `systemd-inhibit` to block idle sleep

This means the screen won't lock or sleep as long as you're sitting there —
even if you don't touch the keyboard. When you walk away, sleep/lock
behavior returns to normal.

Disable with `--no-keep-awake` if you prefer the system's default power settings.

## Auto-Start

### Windows
Run `install.bat` — creates a VBS script in the Startup folder. Starts on next login.

### Linux
Run `install.sh` — creates a `systemd --user` service. Starts on next login.
```bash
systemctl --user start lock-on-absence   # start now
systemctl --user stop lock-on-absence    # stop
```

## Requirements

- Python 3.8+
- A webcam
- opencv-python + opencv-contrib-python (`pip install -r requirements.txt`)

## Tips

- **Re-enroll after updates** — the model format may change between versions
- **Good lighting** improves detection (avoid strong backlight)
- **Profile detection** works best when you turn your head ~45° sideways
- **Keep-awake** replaces the need for "unlock with face" — the screen simply won't lock while you're there
- If it locks too fast, increase `--delay`
- If it doesn't detect you, try `--camera 1`

## Uninstall

**Windows:** delete `LockOnAbsence.vbs` from your Startup folder  
**Linux:** `systemctl --user disable lock-on-absence` + remove the service file

## License

MIT
