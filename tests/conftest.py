"""
Hermetic defaults for the test suite.

Pinning DEFENDER_BACKEND=stub here makes the suite hermetic — even if a CI
runner has DEFENDER_BACKEND=http set globally, tests never reach for the
network.
"""
import os

os.environ.setdefault("DEFENDER_BACKEND", "stub")
