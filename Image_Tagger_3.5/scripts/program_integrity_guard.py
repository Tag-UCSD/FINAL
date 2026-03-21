"""
Program Integrity Guard

Scans for dangerous ellipsis placeholders and uncontrolled stubs in live code.

Rules:
- A bare "..." line is forbidden outside the ellipsis_allowlist.
- "# STUB:" is allowed only in files listed in stub_allowlist.
- Archive directories are ignored.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import yaml  # type: ignore


ROOT = Path(__file__).resolve().parents[1]


def load_release_policy() -> dict:
    cfg_path = ROOT / "release.keep.yml"
    if not cfg_path.exists():
        print("[program_integrity_guard] release.keep.yml not found; treating as NO-GO.", file=sys.stderr)
        raise SystemExit(1)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_source_files() -> Iterable[Path]:
    for root, dirs, files in os.walk(ROOT):
        rel_root = Path(root).relative_to(ROOT)
        # Skip archive, __pycache__, and virtual environments
        if any(part in {"archive", "__pycache__", ".venv", "venv"} for part in rel_root.parts):
            continue
        for name in files:
            if name.endswith(".py"):
                if name in {"program_integrity_guard.py", "syntax_guard.py", "critical_import_guard.py"}:
                    continue
                yield Path(root) / name


def main() -> None:
    policy = load_release_policy()
    stub_allow = set(policy.get("stub_allowlist") or [])
    ellipsis_allow = set(policy.get("ellipsis_allowlist") or [])

    bad_ellipsis: list[tuple[str, int, str]] = []
    bad_stubs: list[tuple[str, int, str]] = []

    for path in iter_source_files():
        rel = str(path.relative_to(ROOT))
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                stripped = line.strip()
                if stripped == "...":
                    if rel not in ellipsis_allow:
                        bad_ellipsis.append((rel, lineno, line.rstrip("\n")))
                if "# STUB:" in line:
                    if rel not in stub_allow:
                        bad_stubs.append((rel, lineno, line.rstrip("\n")))

    if bad_ellipsis or bad_stubs:
        print("[program_integrity_guard] Integrity violations detected:", file=sys.stderr)
        if bad_ellipsis:
            print("  Bare ellipses (forbidden):", file=sys.stderr)
            for rel, lineno, snippet in bad_ellipsis:
                print(f"    {rel}:{lineno}: {snippet}", file=sys.stderr)
        if bad_stubs:
            print("  Unallowlisted stubs:", file=sys.stderr)
            for rel, lineno, snippet in bad_stubs:
                print(f"    {rel}:{lineno}: {snippet}", file=sys.stderr)
        raise SystemExit(1)

    print("[program_integrity_guard] OK: no dangerous ellipses or uncontrolled stubs.")


if __name__ == "__main__":
    main()
