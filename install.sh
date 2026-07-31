#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  install.sh — venv + pip install + systemd user units (agent + watchdog)
#
#  Deliberate choices:
#   * a dedicated venv, because PEP 668 (Ubuntu 23.04+, Debian 12+, Fedora,
#     Arch) makes a bare `pip install` fail and `set -e` would abort mid-way
#   * absolute ExecStart, because systemd rejects a relative executable
#   * Type=notify, because the agent now sends sd_notify(WATCHDOG=1); a
#     Type=simple service with WatchdogSec never notifies and systemd kills
#     and restarts it every interval, forever
#   * autostart is opt-in, asked out loud. A webcam monitor that installs
#     itself silently is indistinguishable from stalkerware
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
UNIT_DIR="$HOME/.config/systemd/user"

echo "lock-on-absence installer"
echo "  repo: $HERE"

command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }
PY_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || { echo "ERROR: Python 3.10+ required (found $PY_VER)"; exit 1; }
echo "  python: $PY_VER"

echo "==> creating venv at $VENV"
python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
echo "==> installing package"
"$PY" -m pip install --quiet -e "$HERE"

AGENT="$VENV/bin/lock-on-absence"
WATCHDOG="$VENV/bin/lock-on-absence-watchdog"
[ -x "$AGENT" ] || { echo "ERROR: entry point missing after install"; exit 1; }
echo "==> installed: $("$AGENT" --version)"

# ── enrollment ────────────────────────────────────────────────────────────
if [ ! -f "$HERE/face_model.yml" ]; then
  cat <<'MSG'

No face model found. Without one the agent refuses to start, on purpose:
running in --any-face mode means ANY face keeps your screen unlocked, which
provides no protection against a person.

Enroll now with:
    .venv/bin/lock-on-absence-enroll

MSG
fi

# ── autostart, opt-in ─────────────────────────────────────────────────────
read -r -p "Enable autostart at login (agent + watchdog)? [y/N] " REPLY
if [[ ! "${REPLY:-}" =~ ^[Yy]$ ]]; then
  echo "==> autostart NOT enabled. Run manually:"
  echo "      $AGENT --delay 10"
  echo "      $WATCHDOG --heartbeat $HERE/watchdog_heartbeat.txt"
  exit 0
fi

mkdir -p "$UNIT_DIR"

cat > "$UNIT_DIR/lock-on-absence.service" <<UNIT
[Unit]
Description=Lock on Absence — webcam presence detection
Documentation=https://github.com/handnewb/lock-on-absence
After=graphical-session.target
PartOf=graphical-session.target

[Service]
# Type=notify pairs with the agent's sd_notify(WATCHDOG=1). Do not change this
# to simple while WatchdogSec is set, or systemd will kill it every 30s.
Type=notify
NotifyAccess=main
WatchdogSec=30
WorkingDirectory=$HERE
ExecStart=$AGENT --delay 10 --mode security
Restart=always
RestartSec=10
# Hardening: this process needs a camera and a session bus, nothing else.
NoNewPrivileges=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes

[Install]
WantedBy=default.target
UNIT

cat > "$UNIT_DIR/lock-on-absence-watchdog.service" <<UNIT
[Unit]
Description=Lock on Absence watchdog — locks if the agent stops beating
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$HERE
ExecStart=$WATCHDOG --heartbeat $HERE/watchdog_heartbeat.txt --max-age 120
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now lock-on-absence.service
systemctl --user enable --now lock-on-absence-watchdog.service

echo
echo "==> autostart ENABLED"
echo "    status:  systemctl --user status lock-on-absence"
echo "    logs:    journalctl --user -u lock-on-absence -f"
echo "    stop:    systemctl --user stop lock-on-absence lock-on-absence-watchdog"
echo "    remove:  systemctl --user disable --now lock-on-absence{,-watchdog}"
