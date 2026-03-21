"""Canonical feature registry guard.

Ensures coverage between the canonical feature registry
(features_canonical.jsonl) and the actual compute implementations.

A feature key is allowed to exist in the registry if and only if it is:
- computed somewhere in backend/science via frame.add_attribute, or
- explicitly listed in backend.science.feature_stubs.STUB_FEATURE_KEYS.

This script is designed to be robust to partial or placeholder lines in the
JSONL file (it will skip lines that fail JSON decoding), so that it fails
only on *true* coverage issues rather than formatting glitches.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Add project root to sys.path so we can import backend modules
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.science import feature_stubs
CANON_PATH = ROOT / "backend" / "science" / "features_canonical.jsonl"


def _load_registry_keys() -> set[str]:
    """Load canonical feature keys from the JSONL registry.

    Each non-empty line is expected to be a JSON object with at least a
    'key' field. We also tolerate escaped newlines ("\\n") in the
    'description' field, similar to the test harness.
    """
    if not CANON_PATH.exists():
        print(
            f"[canon_guard] Canonical registry file not found: {CANON_PATH}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    raw = CANON_PATH.read_text(encoding="utf-8", errors="ignore")
    keys: set[str] = set()

    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        # Some export pipelines store escaped newlines in JSONL; unescape them
        # for compatibility with the test harness behaviour.
        normalized = stripped.replace("\\n", "\n")

        try:
            obj = json.loads(normalized)
        except Exception:
            # Be conservative: skip obviously placeholder or truncated lines in
            # this guard, rather than failing with a JSON error. The program
            # integrity guard is responsible for policing placeholders.
            continue

        key = obj.get("key")
        if isinstance(key, str) and key:
            keys.add(key)

    return keys


def _load_computed_keys() -> set[str]:
    """Scan backend/science for frame.add_attribute("<key>") calls.

    This mirrors the behaviour of tests/test_feature_registry_coverage.py,
    but packaged for CI/install-time use.
    """
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
