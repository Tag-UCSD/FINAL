#!/usr/bin/env python3
"""
THE GUARDIAN - Governance Enforcement Script (v3.2)

Usage:
  python scripts/guardian.py freeze   -> Snapshot current state to governance.lock
  python scripts/guardian.py verify   -> Check current state against governance.lock
"""

import sys
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml  # Requires PyYAML

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "v3_governance.yml"
LOCK_FILE = REPO_ROOT / "governance.lock"


def load_config() -> Dict[str, Any]:
    """Load the governance config from v3_governance.yml."""
    if not CONFIG_FILE.exists():
        print("[guardian] v3_governance.yml not found; using empty config.")
        return {}
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a baseline snapshot:
    - Hashes of all files under protected scopes
    - List of critical files
    - Root-level files (for prevent_new_root_files)
    """
    protected_scopes: List[str] = conf.get("protected_scopes", []) or []
    critical_files: List[str] = conf.get("critical_files", []) or []
    constraints: Dict[str, Any] = conf.get("constraints", {}) or {}

    protected_files: Dict[str, Dict[str, Any]] = {}

    for scope in protected_scopes:
        scope_path = REPO_ROOT / scope
        if not scope_path.exists():
            continue
        if scope_path.is_file():
            rel = scope_path.relative_to(REPO_ROOT).as_posix()
            protected_files[rel] = {
                "hash": sha256_file(scope_path),
                "size": scope_path.stat().st_size,
            }
            continue

        for p in scope_path.rglob("*"):
            if p.is_file():
                rel = p.relative_to(REPO_ROOT).as_posix()
                protected_files[rel] = {
                    "hash": sha256_file(p),
                    "size": p.stat().st_size,
                }

    root_files = sorted([p.name for p in REPO_ROOT.iterdir() if p.is_file()])

    snapshot_obj: Dict[str, Any] = {
        "policy_version": conf.get("policy_version", "3.0.0"),
        "protected_files": protected_files,
        "critical_files": critical_files,
        "constraints": constraints,
        "root_files": root_files,
    }
    return snapshot_obj


def freeze(conf: Dict[str, Any]) -> None:
    """Write a new governance.lock baseline."""
    snap = snapshot(conf)
    with LOCK_FILE.open("w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, sort_keys=True)
    print(f"[guardian] Wrote baseline to {LOCK_FILE}")


def verify(conf: Dict[str, Any]) -> int:
    """
    Verify the current repo state against governance.lock.

    Rules:
    - All critical_files must exist and be >= min_file_size_bytes.
    - All protected_files from the baseline must still exist and not be trivially small.
    - If prevent_new_root_files is true, no new root-level files may appear
      (except governance.lock itself).
    - Hash changes in protected files are treated as failures; they should
      be followed by a manual freeze when intentionally updating the baseline.
    """
    if not LOCK_FILE.exists():
        print("[guardian] No governance.lock baseline; nothing to verify yet.")
        # Treat as success to avoid blocking installs before first freeze.
        return 0

    with LOCK_FILE.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    constraints: Dict[str, Any] = baseline.get("constraints") or conf.get("constraints", {}) or {}
    min_size = int(constraints.get("min_file_size_bytes", 0) or 0)
    prevent_new_root = bool(constraints.get("prevent_new_root_files", False))

    failures: List[str] = []

    # Critical files: existence + minimum size
    critical_files: List[str] = baseline.get("critical_files") or conf.get("critical_files", []) or []
    for rel in critical_files:
        p = REPO_ROOT / rel
        if not p.exists():
            failures.append(f"Critical file missing: {rel}")
        else:
            size = p.stat().st_size
            if size < min_size:
                failures.append(f"Critical file too small: {rel} ({size} bytes)")

    # Protected files: existence + minimum size + hash stability
    protected_files: Dict[str, Dict[str, Any]] = baseline.get("protected_files") or {}
    for rel, info in protected_files.items():
        p = REPO_ROOT / rel
        if not p.exists():
            failures.append(f"Protected file missing: {rel}")
            continue

        size = p.stat().st_size
        if size < min_size:
            failures.append(f"Protected file unexpectedly small: {rel} ({size} bytes)")

        old_hash = info.get("hash")
        if old_hash:
            new_hash = sha256_file(p)
            if new_hash != old_hash:
                failures.append(f"Protected file hash changed: {rel}")

    # Root-level file drift
    if prevent_new_root:
        baseline_root_files = set(baseline.get("root_files") or [])
        current_root_files = {p.name for p in REPO_ROOT.iterdir() if p.is_file()}
        extras = sorted(current_root_files - baseline_root_files)
        # governance.lock is expected and safe as a new root file
        extras = [name for name in extras if name not in {"governance.lock"}]
        if extras:
            failures.append(
                "New root-level files detected: " + ", ".join(extras)
            )

    if failures:
        print("[guardian] VERIFICATION FAILED:")
        for msg in failures:
            print(" -", msg)
        sys.exit(1)
    else:
        print("[guardian] Verification OK.")
        return 0


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/guardian.py [freeze|verify]")
        return 1

    mode = argv[1]
    conf = load_config()

    if mode == "freeze":
        freeze(conf)
        return 0
    elif mode == "verify":
        return verify(conf)
    else:
        print(f"Unknown mode: {mode}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))