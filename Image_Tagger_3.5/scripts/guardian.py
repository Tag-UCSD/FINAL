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
from typing import Any, Dict, List, Optional

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


def freeze(conf: Dict[str, Any], lock_path: Optional[Path] = None) -> None:
    """Create or refresh the governance.lock baseline.

    Parameters
    ----------
    conf:
        Parsed config from v3_governance.yml.
    lock_path:
        Optional override for the lock file location. When omitted, the
        global LOCK_FILE is used. Tests may supply a temporary path to
        avoid mutating the real governance.lock.
    """
    if lock_path is None:
        lock_path = LOCK_FILE

    baseline = snapshot(conf)
    lock_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    rel = lock_path.relative_to(REPO_ROOT).as_posix()
    print(f"[guardian] Baseline written to {rel}")



def _check_science_tag_coverage(constraints: Dict[str, Any], failures: List[str]) -> None:
    """Enforce that the science tag coverage has no 'unassigned' keys.

    When `constraints.enforce_science_tag_coverage` is true, this function
    expects a `science_tag_coverage_v1.json` file at the repo root and will
    fail verification if any feature keys are reported with source_type
    "unassigned".
    """
    enforce = bool(constraints.get("enforce_science_tag_coverage", False))
    if not enforce:
        return

    cov_path = REPO_ROOT / "science_tag_coverage_v1.json"
    if not cov_path.exists():
        failures.append(
            "Science tag coverage: science_tag_coverage_v1.json is missing; "
            "run `python scripts/generate_tag_coverage.py` before verifying."
        )
        return

    try:
        payload = json.loads(cov_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        failures.append(f"Science tag coverage: failed to read JSON: {exc}")
        return

    meta = payload.get("meta") or {}
    counts = meta.get("counts_by_source_type") or {}
    unassigned = int(counts.get("unassigned", 0) or 0)
    if unassigned > 0:
        failures.append(
            f"Science tag coverage: {unassigned} feature key(s) are 'unassigned'. "
            "Every key must be wired, explicitly stubbed, or removed from the registry."
        )


def _check_bn_db_health(constraints: Dict[str, Any], failures: List[str]) -> None:
    """Optionally run BN / DB health checks via backend.scripts.bn_db_health.

    When `constraints.check_bn_db_health` is true, this will attempt to run
    the BN / DB health checker in-process and append a human-readable
    failure message if any violations are detected.

    This check requires a running database. If the checker cannot be
    imported or raises an exception, the error is recorded as a failure.
    """
    if not constraints.get("check_bn_db_health"):
        return

    try:
        from backend.scripts import bn_db_health
    except Exception as exc:  # pragma: no cover - import errors
        failures.append(f"BN/DB health: could not import checker: {exc}")
        return

    try:
        summary = bn_db_health.run_health_check(exit_on_failure=False)
    except Exception as exc:  # pragma: no cover - runtime errors
        failures.append(f"BN/DB health: check raised an exception: {exc}")
        return

    if not summary.get("ok", False):
        orphan = summary.get("orphan_validations", 0)
        missing = summary.get("missing_candidates", 0)
        failures.append(
            "BN/DB health: "
            f"{orphan} orphan Validation.attribute_key rows and "
            f"{missing} missing BN candidate keys in attributes; "
            "run `python -m backend.scripts.bn_db_health` for details."
        )

def verify(conf: Dict[str, Any], lock_path: Optional[Path] = None) -> int:
    """Verify the current repo state against governance.lock.

    Rules
    -----
    - All critical_files must exist and be >= min_file_size_bytes.
    - All protected_files from the baseline must still exist and not be trivially small.
    - If prevent_new_root_files is true, no new root-level files may appear
      (except governance.lock itself).
    - Hash changes in protected files are treated as failures; they should
      be followed by a manual freeze when intentionally updating the baseline.
    """
    if lock_path is None:
        lock_path = LOCK_FILE

    if not lock_path.exists():
        print("[guardian] No governance.lock baseline; nothing to verify yet.")
        # Treat as success to avoid blocking installs before first freeze.
        return 0

    try:
        baseline = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[guardian] Failed to read {lock_path}: {exc}")
        return 1

    critical_files = baseline.get("critical_files") or []
    protected_files: Dict[str, Dict[str, Any]] = baseline.get("protected_files") or {}
    root_files = set(baseline.get("root_files") or [])

    constraints = conf.get("constraints", {}) or {}
    min_size = int(constraints.get("min_file_size_bytes", 0))
    prevent_new_root = bool(constraints.get("prevent_new_root_files", False))

    failures = []

    # Science tag coverage enforcement (no 'unassigned' feature keys)
    _check_science_tag_coverage(constraints, failures)

    # Optional BN / DB health check (requires a running database)
    _check_bn_db_health(constraints, failures)

    # Critical files: existence + minimum size
    for rel in critical_files:
        p = REPO_ROOT / rel
        if not p.exists():
            failures.append(f"Critical file missing: {rel}")
        else:
            size = p.stat().st_size
            if size < min_size:
                failures.append(f"Critical file too small: {rel} ({size} bytes)")

    # Protected files: existence + minimum size + hash stability
    for rel, info in protected_files.items():
        p = REPO_ROOT / rel
        if not p.exists():
            failures.append(f"Protected file missing: {rel}")
            continue

        size = p.stat().st_size
        if size < min_size:
            failures.append(f"Protected file too small: {rel} ({size} bytes)")
            continue

        old_hash = info.get("hash")
        new_hash = sha256_file(p)
        if old_hash and new_hash != old_hash:
            failures.append(f"Protected file hash changed: {rel}")

    # Root-level file drift
    if prevent_new_root:
        baseline_root_files = set(root_files)
        current_root_files = {p.name for p in REPO_ROOT.iterdir() if p.is_file()}
        extras = sorted(current_root_files - baseline_root_files)
        # governance.lock is expected and safe as a new root file
        extras = [name for name in extras if name not in {"governance.lock"}]
        if extras:
            failures.append(
                "New root-level files detected: " + ", ".join(extras)
            )

    if failures:
        print("[guardian] Verification FAILED:")
        for msg in failures:
            print(" -", msg)
        return 1

    print("[guardian] Verification PASSED.")
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