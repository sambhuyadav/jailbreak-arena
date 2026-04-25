"""
Environment-level configuration. Read once at import time.

All env vars listed here are environment / arena scoped. Defender-scoped vars
(DEFENDER_*) live in defender.py.
"""
import os

MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))
if MAX_TURNS < 1:
    raise ValueError(f"MAX_TURNS must be >= 1, got {MAX_TURNS}")
