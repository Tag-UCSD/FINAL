"""
Centralized version helper.

The VERSION file at the repository root is the single source of truth for
the Image Tagger version string. Other components should import VERSION
from this module instead of hard-coding a value.
"""

from pathlib import Path


def _read_version() -> str:
    """
    Return the current Image Tagger version string.

    We resolve the repository root by going one directory up from the
    backend/ package and looking for a VERSION file there.

    If anything goes wrong (e.g. the file is missing inside a container),
    we fall back to "dev".
    """
    try:
        root = Path(__file__).resolve().parents[1]
        version_file = root / "VERSION"
        return version_file.read_text(encoding="utf-8").strip()
    except Exception:
        return "dev"


VERSION: str = _read_version()
