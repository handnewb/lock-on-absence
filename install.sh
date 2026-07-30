#!/usr/bin/env bash
# Lock on Absence — Linux / macOS installer
# Installs dependencies, tests webcam, and optionally sets up auto-start.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  Lock on Absence — Linux/macOS Installer"
echo "================================================"
echo ""

# 1. Python check
echo "[1/4] Checking Python..."
PYTHON=""
for p in python3 python; do
    if command -v "$p" &>/dev/null; then
        PYTHON="$p"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Install with your package manager."
    exit 1
fi
$PYTHON --version
echo ""

# 2. Dependencies (venv to avoid PEP 668)
echo "[2/4] Installing dependencies..."
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install -r requirements.txt --quiet 2>/dev/null || \
    "$VENV_DIR/bin/pip" install -r requirements.txt --user --quiet
echo "OK"
echo ""

PYTHON_ABS="$VENV_DIR/bin/python"

# 3. Webcam test
echo "[3/4] Testing webcam..."
$PYTHON_ABS -c "
import cv2
found = False
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Webcam {i}: OK')
        found = True
    cap.release()
if not found:
    print('WARNING: no webcam found')
"
echo ""

# 4. Auto-start (systemd user service)
echo "[4/4] Setting up auto-start..."
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_DIR/lock-on-absence.service" << EOF
[Unit]
Description=Lock on Absence — auto screen lock
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=$PYTHON_ABS $SCRIPT_DIR/lock-on-absence.py
WorkingDirectory=$SCRIPT_DIR
Restart=always
RestartSec=10
# Watchdog disabled by default — enable via override if sd_notify is implemented

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload 2>/dev/null || true
systemctl --user enable lock-on-absence.service 2>/dev/null || true

echo ""
echo "================================================"
echo "  Done!"
echo ""
echo "  To enroll your face (recommended):"
echo "    python3 enroll.py"
echo ""
echo "  To start now:"
echo "    python3 lock-on-absence.py"
echo "    # or: systemctl --user start lock-on-absence"
echo ""
echo "  Auto-start: ENABLED (systemd user service)"
echo ""
echo "  To UNINSTALL:"
echo "    systemctl --user disable lock-on-absence"
echo "    rm $SERVICE_DIR/lock-on-absence.service"
echo "================================================"
