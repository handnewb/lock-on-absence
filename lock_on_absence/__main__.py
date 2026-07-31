"""Allow `python -m lock_on_absence` to run the agent."""
import sys

from .agent import main

if __name__ == "__main__":
    sys.exit(main())
