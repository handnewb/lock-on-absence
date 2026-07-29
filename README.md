# Lock on Absence

Auto-lock your screen when you walk away from your computer — using just your webcam.

**Two modes:**

| Mode | Behavior |
|------|----------|
| **Any face** (default) | Locks when NO face is detected. Any person in front keeps it unlocked. |
| **Owner recognition** | Locks when YOUR face is NOT detected. A different person triggers the lock too. |

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
2. Detects faces with OpenCV Haar Cascade
3. If owner recognition is enabled (model exists), checks whether the face matches
4. When the owner is gone for N seconds → locks the screen

On **Windows**: calls `LockWorkStation()`  
On **Linux**: calls `loginctl lock-session` (falls back to xdg-screensaver, i3lock, slock)  
On **macOS**: puts display to sleep

## CLI Options

```
python lock-on-absence.py --help

  --delay 5            Seconds without owner before lock (default: 10)
  --camera 1           Use camera index 1
  --no-lock            Dry-run: detect but never lock
  --check-interval 2   Seconds between checks (default: 1.5)
  --model face.yml     Path to trained model
```

## Enrolling Your Face

```bash
python enroll.py                   # 30 samples (~15s)
python enroll.py --samples 50      # 50 samples, more accurate
```

Move your head slowly: **front → left → right → up → down**.

The model is saved as `face_model.yml` in the script directory.

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

- **Good lighting** improves detection (avoid strong backlight)
- **Remove glasses/hats** during enrollment for best results
- **Recognition mode** requires enrollment first (`python enroll.py`)
- If it locks too fast, increase `--delay`
- If it doesn't detect you, try `--camera 1`

## Uninstall

**Windows:** delete `LockOnAbsence.vbs` from your Startup folder  
**Linux:** `systemctl --user disable lock-on-absence` + remove the service file

## License

MIT
