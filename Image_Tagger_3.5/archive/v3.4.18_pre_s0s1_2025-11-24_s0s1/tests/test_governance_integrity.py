"""
Regression tests for governance guard scripts.

These are intentionally light-weight: they just assert that the guard
scripts run without raising SystemExit under the current tree.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hollow_repo_guard_runs():
    runpy.run_path(str(ROOT / "scripts" / "hollow_repo_guard.py"))


def test_program_integrity_guard_runs():
    runpy.run_path(str(ROOT / "scripts" / "program_integrity_guard.py"))
