"""
Watchdog — external process that locks the screen if the agent's heartbeat
goes stale.  Run this as a scheduled task or cron job.

Heartbeat file: lock-on-absence writes the current timestamp to
`watchdog_heartbeat.txt` every 60s when owner is present.

This script reads the file; if the timestamp is older than MAX_AGE,
it locks the screen.  This catches crashes, zombs, and kills.
"""
import os
import sys
import time
import ctypes

HEARTBEAT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "watchdog_heartbeat.txt")
MAX_AGE = 120  # seconds — how old can heartbeat be before triggering lock?
CHECK_INTERVAL = 30  # seconds between checks


def lock_now() -> None:
    """Lock the workstation immediately."""
    if sys.platform == "win32":
        ctypes.windll.user32.LockWorkStation()
    else:
        import subprocess
        for args in (
            ["loginctl", "lock-session"],
            ["xdg-screensaver", "lock"],
            ["gnome-screensaver-command", "--lock"],
        ):
            try:
                r = subprocess.run(args, timeout=5, check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if r.returncode == 0:
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue


def main() -> None:
    while True:
        try:
            with open(HEARTBEAT_FILE) as f:
                ts_str = f.read().strip()
            last_beat = float(ts_str)
            age = time.time() - last_beat
            if age > MAX_AGE:
                print(f"Heartbeat stale ({age:.0f}s) — locking screen")
                lock_now()
        except (FileNotFoundError, ValueError):
            # No heartbeat file yet — agent probably not started
            pass
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
