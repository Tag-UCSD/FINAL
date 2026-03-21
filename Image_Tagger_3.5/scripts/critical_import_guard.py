"""Critical Import Guard

Attempts to import critical local modules to ensure they can be loaded
without local-code failures. Third-party missing dependencies are treated
as warnings (install/docker will handle them), but local package failures
are hard NO-GO.
"""

from __future__ import annotations

import importlib
import sys
from typing import List, Tuple

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CRITICAL_MODULES = [
    "backend.science.pipeline",
    "backend.main",
]


def _is_third_party_missing(e: ModuleNotFoundError) -> bool:
    name = getattr(e, "name", None)
    return bool(name) and not name.startswith("backend") and not name.startswith("scripts")


def main() -> None:
    failures: List[Tuple[str, str]] = []
    warnings: List[Tuple[str, str]] = []

    for mod in CRITICAL_MODULES:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if _is_third_party_missing(e):
                warnings.append((mod, f"missing third-party dependency: {e.name}"))
            else:
                failures.append((mod, f"missing local dependency: {e}"))
        except Exception as e:
            failures.append((mod, f"import failed: {type(e).__name__}: {e}"))

    if warnings:
        print("[critical_import_guard] WARNINGS:")
        for mod, msg in warnings:
            print(f"  {mod}: {msg}")

    if failures:
        print("[critical_import_guard] NO-GO", file=sys.stderr)
        for mod, msg in failures:
            print(f"  {mod}: {msg}", file=sys.stderr)
        raise SystemExit(1)

    print("[critical_import_guard] OK: critical modules import cleanly.")


if __name__ == "__main__":
    main()
