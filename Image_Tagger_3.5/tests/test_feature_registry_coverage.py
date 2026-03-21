"""
Tests ensuring coverage between the canonical feature registry
and the actual compute implementations.

A feature key is allowed to exist in the registry if and only if it is:
- computed somewhere in backend/science via frame.add_attribute, or
- explicitly listed in backend.science.feature_stubs.STUB_FEATURE_KEYS.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.science import feature_stubs


ROOT = Path(__file__).resolve().parents[1]


def _load_registry_keys() -> set[str]:
    """Parse backend/science/features_canonical.jsonl into a set of keys.

    The file is currently stored as JSON lines with escaped newlines ("\n"),
    so we first unescape those before splitting.
    """
    reg_path = ROOT / "backend" / "science" / "features_canonical.jsonl"
    text = reg_path.read_text(encoding="utf-8")
    # Convert "\n" sequences into real newlines so each object is on its own line.
    text = text.replace("\\n", "\n")
    keys: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        key = obj.get("key")
        if key:
            keys.add(key)
    return keys


def _load_computed_keys() -> set[str]:
    """Scan backend/science for add_attribute() calls and extract keys."""
    base = ROOT / "backend" / "science"
    keys: set[str] = set()
    for path in base.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r'add_attribute\("([^"]+)"', text):
            keys.add(m.group(1))
    return keys


def test_registry_keys_are_covered_by_compute_or_stub() -> None:
    registry_keys = _load_registry_keys()
    computed_keys = _load_computed_keys()
    stub_keys = feature_stubs.STUB_FEATURE_KEYS

    dangling = registry_keys - computed_keys - stub_keys
    assert not dangling, f"Dangling registry keys with no compute or stub: {sorted(dangling)[:20]}"
