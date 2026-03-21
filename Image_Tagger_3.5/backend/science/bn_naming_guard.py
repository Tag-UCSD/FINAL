"""Lightweight BN naming guard for v3.4.36.

This script performs a best-effort scan of Bayesian Network configuration
files under ``backend/science`` and prints a summary of node names and
potential naming issues (e.g., whitespace in node identifiers).

It is deliberately *non-fatal*: it never exits with a non-zero status so it
is safe to run inside GO checks and ``install.sh`` without blocking student
workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


ROOT = Path(__file__).resolve().parent


def _iter_bn_files() -> Iterable[Path]:
    """Yield BN-related config files under ``backend/science``.

    We look for JSON / YAML files whose names contain ``bn`` or ``rest``.
    """
    patterns = ("**/*.json", "**/*.yml", "**/*.yaml")
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            name = path.name.lower()
            if "bn" in name or "rest" in name:
                yield path


def _extract_node_names(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []

    data = None
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except Exception:
            return []
    else:
        try:
            import yaml  # type: ignore
        except Exception:
            return []
        try:
            data = yaml.safe_load(text)
        except Exception:
            return []

    names: List[str] = []
    if isinstance(data, dict):
        if isinstance(data.get("nodes"), list):
            for node in data["nodes"]:
                if isinstance(node, dict) and "name" in node:
                    names.append(str(node["name"]))
        if isinstance(data.get("variables"), dict):
            names.extend(str(k) for k in data["variables"].keys())
    return names


def main() -> None:
    all_names: Dict[Path, List[str]] = {}
    for path in sorted(set(_iter_bn_files())):
        names = _extract_node_names(path)
        if names:
            all_names[path] = sorted(set(names))

    if not all_names:
        print("[bn_naming_guard] No BN config files detected; nothing to validate.")
        return

    print("[bn_naming_guard] Summary of BN node names:")
    problematic: Dict[Path, List[str]] = {}
    for path, names in all_names.items():
        bad = [n for n in names if any(c.isspace() for c in n)]
        print(f"  - {path.relative_to(ROOT)}: {len(names)} nodes")
        if bad:
            problematic[path] = bad

    if problematic:
        print("\n[bn_naming_guard] Potential naming issues (whitespace in names):")
        for path, bad in problematic.items():
            rel = path.relative_to(ROOT)
            joined = ", ".join(sorted(bad))
            print(f"  - {rel}: {joined}")
    else:
        print("\n[bn_naming_guard] No obvious naming issues detected.")

    print("\n[bn_naming_guard] Completed without fatal errors.")


if __name__ == "__main__":
    main()
