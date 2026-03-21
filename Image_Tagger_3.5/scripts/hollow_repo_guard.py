"""
Hollow Repo Guard

Simple structural sanity check to prevent "shell" releases.

Checks:
- critical_paths exist
- min_counts thresholds (per release.keep.yml) are respected
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml  # type: ignore


ROOT = Path(__file__).resolve().parents[1]


def load_release_policy() -> dict:
    cfg_path = ROOT / "release.keep.yml"
    if not cfg_path.exists():
        print("[hollow_repo_guard] release.keep.yml not found; treating as NO-GO.", file=sys.stderr)
        raise SystemExit(1)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_critical_paths(policy: dict) -> list[str]:
    missing = []
    for rel in policy.get("critical_paths", []):
        target = ROOT / rel
        if not target.exists():
            missing.append(rel)
    return missing


def count_files_under(rel: str) -> int:
    base = ROOT / rel
    if not base.exists():
        return 0
    total = 0
    for root, dirs, files in os.walk(base):
        total += len(files)
    return total


def check_min_counts(policy: dict) -> dict[str, int]:
    violations: dict[str, int] = {}
    for rel, min_count in (policy.get("min_counts") or {}).items():
        actual = count_files_under(rel)
        if actual < int(min_count):
            violations[rel] = actual
    return violations


def main() -> None:
    policy = load_release_policy()
    missing = check_critical_paths(policy)
    if missing:
        print("[hollow_repo_guard] Missing critical paths:", file=sys.stderr)
        for rel in missing:
            print(f"  - {rel}", file=sys.stderr)
        raise SystemExit(1)

    violations = check_min_counts(policy)
    if violations:
        print("[hollow_repo_guard] Min-count violations:", file=sys.stderr)
        for rel, actual in violations.items():
            print(f"  - {rel}: {actual} files (below configured minimum)", file=sys.stderr)
        raise SystemExit(1)

    print("[hollow_repo_guard] OK: structure looks non-hollow.")


if __name__ == "__main__":
    main()
