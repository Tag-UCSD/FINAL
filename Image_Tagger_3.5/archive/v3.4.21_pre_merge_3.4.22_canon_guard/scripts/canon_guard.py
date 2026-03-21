"""Canonical feature registry guard.

Mirrors the logic of tests/test_feature_registry_coverage.py but is
packaged as a standalone script suitable for CI/install-time checks.

A feature key is allowed to exist in the registry if and only if it is:
- computed somewhere in backend/science via frame.add_attribute, or
- explicitly listed in backend.science.feature_stubs.STUB_FEATURE_KEYS.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from backend.science import feature_stubs

ROOT = Path(__file__).resolve().parents[1]


def _load_registry_keys() -> set[str]:
    registry_path = ROOT / "backend" / "science" / "feature_registry.json"
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return set(data.keys())


def _load_computed_keys() -> set[str]:
    base = ROOT / "backend" / "science"
    keys: set[str] = set()
    for path in base.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r'add_attribute\("([^"]+)"', text):
            keys.add(m.group(1))
    return keys


def main() -> None:
    registry_keys = _load_registry_keys()
    computed_keys = _load_computed_keys()
    stub_keys = feature_stubs.STUB_FEATURE_KEYS

    dangling = registry_keys - computed_keys - stub_keys
    if dangling:
        print("[canon_guard] Dangling registry keys with no compute or stub:", file=sys.stderr)
        for key in sorted(dangling)[:20]:
            print(f"  - {key}", file=sys.stderr)
        raise SystemExit(1)

    print("[canon_guard] OK: registry keys covered by compute or stubs.")

if __name__ == "__main__":
    main()
