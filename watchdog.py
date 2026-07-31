#!/usr/bin/env python3
"""Backward-compatible shim. Real code lives in lock_on_absence/watchdog.py

Kept so existing install.bat, install.sh, the systemd unit and every README
command keep working after the package restructure. Prefer the installed
entry points (`lock-on-absence`, `lock-on-absence-enroll`, ...).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lock_on_absence.watchdog import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
