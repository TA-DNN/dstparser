"""Shared test helpers.

These are integration tests: they read real DST files from shared storage. When
that storage is not mounted the tests skip rather than fail, so `pytest` stays
usable off the analysis machine.
"""

from pathlib import Path

import pytest


def require(*paths):
    """Skip the test unless every path exists."""
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"input not available: {p}")
