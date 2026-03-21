"""Guard script: assert that no __pycache__ directories are present in the repo tree.

This is used in CI to prevent Python bytecode caches from creeping into release artifacts.
"""
from pathlib import Path
import sys

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    bad_paths = []
    for path in root.rglob("__pycache__"):
        if path.is_dir():
            bad_paths.append(path)

    if not bad_paths:
        print("[pycache-guard] OK: no __pycache__ directories found.")
        return 0

    print("[pycache-guard] ERROR: __pycache__ directories found in repo:")
    for p in bad_paths:
        print(f"  - {p}")
    print("[pycache-guard] Please remove these before creating a release ZIP/TXT.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
