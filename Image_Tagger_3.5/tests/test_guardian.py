import json
from pathlib import Path

import pytest

from scripts import guardian


def test_guardian_freeze_and_verify(tmp_path: Path):
    """Guardian freeze+verify round-trip should succeed with a temp lock file."""
    conf = guardian.load_config()
    if not conf:
        pytest.skip("No governance config loaded; skipping Guardian test.")

    temp_lock = tmp_path / "governance.lock"

    # Create a baseline snapshot in a temporary lock file
    guardian.freeze(conf, lock_path=temp_lock)

    # Verify against the same baseline; should pass
    rc_ok = guardian.verify(conf, lock_path=temp_lock)
    assert rc_ok == 0


def test_guardian_detects_hash_mismatch(tmp_path: Path):
    """Guardian should fail if baseline hashes are corrupted in the temp lock file.

    This test ONLY perturbs the baseline JSON written to a temporary lock
    file; it never touches the actual repository files or governance.lock.
    """
    conf = guardian.load_config()
    if not conf:
        pytest.skip("No governance config loaded; skipping Guardian test.")

    temp_lock = tmp_path / "governance_corrupt.lock"

    # Build a clean snapshot
    baseline = guardian.snapshot(conf)
    protected_files = baseline.get("protected_files") or {}
    if not protected_files:
        pytest.skip("No protected files configured; nothing to test.")

    # Corrupt the hash of one protected file in the baseline
    key = next(iter(protected_files.keys()))
    protected_files[key]["hash"] = "0" * 64  # impossible SHA256
    temp_lock.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    rc = guardian.verify(conf, lock_path=temp_lock)
    assert rc == 1
