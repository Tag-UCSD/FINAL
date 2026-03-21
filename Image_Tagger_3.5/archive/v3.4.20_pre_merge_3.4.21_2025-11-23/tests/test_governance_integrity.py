"""Regression tests for governance guard scripts.

These tests assert that all guard scripts *execute* successfully
under the current repo tree.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_guard(script_name: str):
    ns = runpy.run_path(str(ROOT / "scripts" / script_name))
    if "main" in ns:
        ns["main"]()


def test_hollow_repo_guard_runs():
    _run_guard("hollow_repo_guard.py")


def test_program_integrity_guard_runs():
    _run_guard("program_integrity_guard.py")


def test_syntax_guard_runs():
    _run_guard("syntax_guard.py")


def test_critical_import_guard_runs():
    _run_guard("critical_import_guard.py")
