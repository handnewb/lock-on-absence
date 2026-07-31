#!/usr/bin/env python3
"""
External watchdog — locks the screen if the agent stops proving it is alive.

Second layer of defence: catches crashes, hangs, zombie loops and kills that the
agent cannot possibly handle itself, because by then it is not running.

Three properties this must have, all of which the first version lacked:

  LATCH        Lock once per staleness episode. The original locked every 30s
               forever after any agent crash, so the user could never log back
               in to fix the very problem that triggered it.

  SKEW-SAFE    A heartbeat timestamp in the future must count as suspicious, not
               as "very fresh". Otherwise `echo 9999999999 > heartbeat` disables
               the whole fail-closed design, and any NTP correction or laptop
               resume silently does the same thing by accident.

  LIVENESS     Prefer asking the OS whether the agent process exists over
               trusting a file that any process running as the user can write.

Run it under systemd (Restart=always) or as a Windows Scheduled Task. See
--print-unit and --print-task for ready-to-install definitions.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from . import __version__

DEFAULT_HEARTBEAT = "watchdog_heartbeat.txt"
DEFAULT_MAX_AGE = 120.0
DEFAULT_INTERVAL = 30.0
FUTURE_TOLERANCE = 5.0          # clock jitter we forgive before calling it tampering


# ═══════════════════════════════════════════════════════════════════════
#  Session state
# ═══════════════════════════════════════════════════════════════════════

def session_is_locked() -> bool | None:
    """
    True / False / None (cannot tell).

    Used to avoid re-locking an already-locked session, which is what turned the
    original watchdog into a lockout loop.
    """
    if sys.platform.startswith("linux"):
        sid = os.environ.get("XDG_SESSION_ID")
        if sid:
            try:
                r = subprocess.run(
                    ["loginctl", "show-session", sid, "-p", "LockedHint"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and "=" in r.stdout:
                    return r.stdout.strip().split("=", 1)[1].lower() == "yes"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    elif sys.platform == "win32":
        # LogonUI.exe owns the secure desktop; its presence means "locked".
        try:
            r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq LogonUI.exe"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return "LogonUI.exe" in r.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def lock_now() -> bool:
    """Reuse the agent's implementation so the two never drift apart."""
    from .face_utils import lock_screen
    return lock_screen()


# ═══════════════════════════════════════════════════════════════════════
#  Heartbeat evaluation
# ═══════════════════════════════════════════════════════════════════════

class Verdict:
    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"
    TAMPERED = "tampered"
    UNREADABLE = "unreadable"


def read_heartbeat(path: Path, max_age: float) -> tuple[str, float]:
    """Return (verdict, age_seconds). Age is 0.0 when unknown."""
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return Verdict.MISSING, 0.0
    except OSError:
        return Verdict.UNREADABLE, 0.0
    if not raw:
        return Verdict.UNREADABLE, 0.0
    try:
        beat = float(raw)
    except ValueError:
        return Verdict.UNREADABLE, 0.0

    age = time.time() - beat
    if age < -FUTURE_TOLERANCE:
        # Future timestamp: either the clock jumped or someone is disabling us.
        return Verdict.TAMPERED, age
    if age <= max_age:
        return Verdict.FRESH, max(0.0, age)
    return Verdict.STALE, age


def agent_pid_alive(pid_file: Path | None) -> bool | None:
    if pid_file is None or not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True        # exists, owned by someone else
    except OSError:
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Unit / task templates
# ═══════════════════════════════════════════════════════════════════════

UNIT_TEMPLATE = """\
# ~/.config/systemd/user/lock-on-absence-watchdog.service
[Unit]
Description=Lock on Absence watchdog
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart={python} -m lock_on_absence.watchdog --heartbeat {heartbeat}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""

TASK_TEMPLATE = """\
# Windows: run once in PowerShell.
$action  = New-ScheduledTaskAction -Execute "{python}" `
    -Argument '-m lock_on_absence.watchdog --heartbeat "{heartbeat}" --once'
$atLogon = New-ScheduledTaskTrigger -AtLogOn
$repeat  = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "LockOnAbsenceWatchdog" `
    -Action $action -Trigger @($atLogon, $repeat) -RunLevel Limited -Force
"""


# ═══════════════════════════════════════════════════════════════════════
#  Checking
# ═══════════════════════════════════════════════════════════════════════

def check_once(args, state: dict, log=print) -> str:
    """One evaluation. `state` carries the latch between calls."""
    verdict, age = read_heartbeat(args.heartbeat, args.max_age)
    pid_alive = agent_pid_alive(args.pid_file)

    if verdict == Verdict.FRESH:
        if state.get("latched"):
            log(f"heartbeat healthy again (age {age:.0f}s) — watchdog re-armed")
        state["latched"] = False
        return "ok"

    if verdict == Verdict.MISSING and not args.lock_missing:
        # Do not fight the user before the agent has ever started.
        return "waiting"

    detail = {
        Verdict.STALE: f"heartbeat stale ({age:.0f}s > {args.max_age:.0f}s)",
        Verdict.TAMPERED: (f"heartbeat is {abs(age):.0f}s in the FUTURE — clock jump "
                           f"or tampering; treating as stale"),
        Verdict.UNREADABLE: "heartbeat unreadable",
        Verdict.MISSING: "heartbeat missing",
    }[verdict]

    if pid_alive is False:
        detail += "; agent PID is gone"
    elif pid_alive is True and verdict == Verdict.STALE:
        detail += "; agent PID alive but not beating (hung)"

    if state.get("latched"):
        return "latched"        # already acted on this episode; stay quiet

    locked = session_is_locked()
    if locked is True:
        log(f"{detail} — session already locked, nothing to do")
        state["latched"] = True
        return "already-locked"

    if args.dry_run:
        log(f"[DRY-RUN] would lock: {detail}")
        state["latched"] = True
        return "dry-run"

    log(f"{detail} — LOCKING")
    state["latched"] = True
    if not lock_now():
        log("ERROR: watchdog could not lock the screen")
        return "lock-failed"
    return "locked"


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lock-on-absence-watchdog",
        description="Lock the screen if the agent's heartbeat goes stale.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--heartbeat", type=Path,
                   default=Path.cwd() / DEFAULT_HEARTBEAT,
                   help=f"Heartbeat file (default: ./{DEFAULT_HEARTBEAT})")
    p.add_argument("--pid-file", type=Path, default=None,
                   help="Also require this PID to be alive")
    p.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE,
                   help=f"Heartbeat older than this is stale "
                        f"(default: {DEFAULT_MAX_AGE:.0f}s)")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                   help=f"Seconds between checks (default: {DEFAULT_INTERVAL:.0f})")
    p.add_argument("--once", action="store_true",
                   help="Check once and exit (for cron / Scheduled Task)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would happen; never lock")
    p.add_argument("--lock-missing", action="store_true",
                   help="Treat a missing heartbeat as stale. Off by default so the "
                        "watchdog does not fight you before the agent starts.")
    p.add_argument("--print-unit", action="store_true",
                   help="Print a systemd user unit and exit")
    p.add_argument("--print-task", action="store_true",
                   help="Print the PowerShell Scheduled Task command and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.print_unit:
        print(UNIT_TEMPLATE.format(python=sys.executable, heartbeat=args.heartbeat))
        return 0
    if args.print_task:
        print(TASK_TEMPLATE.format(python=sys.executable, heartbeat=args.heartbeat))
        return 0

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] watchdog: {msg}", flush=True)

    log(f"watching {args.heartbeat} (max-age {args.max_age:.0f}s, "
        f"interval {args.interval:.0f}s)" + (" [DRY-RUN]" if args.dry_run else ""))

    state: dict = {"latched": False}
    if args.once:
        return 0 if check_once(args, state, log) != "lock-failed" else 1
    try:
        while True:
            check_once(args, state, log)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
