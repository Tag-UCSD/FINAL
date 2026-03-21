"""backend.services.storage

Image storage root discovery and path normalization.

This module centralizes the IMAGE_STORAGE_ROOT behavior so that:
  * Admin uploads, debug endpoints, and API URL builders agree on where images live.
  * The /static mount can safely expose only the image store.

Storage paths in the DB may be:
  1) absolute paths (legacy tests, manual seeding),
  2) relative paths already under the store (e.g., 'data_store/<uuid>.jpg'),
  3) bare filenames (preferred new behavior).

Public helpers:
  - get_image_storage_root(): Path to store (mkdir'ed).
  - resolve_image_path(storage_path): best-effort filesystem resolution.
  - to_static_path(storage_path): relative path used for /static/<...> URLs.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_IMAGE_STORAGE_ROOT = "data_store"

def get_image_storage_root() -> Path:
    """Return the image storage root, ensuring it exists."""
    root = os.getenv("IMAGE_STORAGE_ROOT", DEFAULT_IMAGE_STORAGE_ROOT)
    p = Path(root)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If mkdir fails (e.g., read-only FS), still return the path.
        pass
    return p


def resolve_image_path(storage_path: str) -> Path:
    """Resolve a DB storage_path to a real file on disk if possible."""
    raw = Path(storage_path)
    if raw.is_file():
        return raw

    root = get_image_storage_root()
    # If storage_path already includes the root prefix (relative)
    try:
        sp = str(raw).lstrip("/")
        root_str = str(root).strip("/")
        if sp.startswith(root_str + "/"):
            candidate = Path(sp)
            if candidate.is_file():
                return candidate
    except Exception:
        pass

    # Try root / raw.name (covers 'data_store/<name>' and absolute paths)
    candidate = root / raw.name
    if candidate.is_file():
        return candidate

    # Try root / raw (covers bare relative paths)
    candidate = root / raw
    if candidate.is_file():
        return candidate

    return raw  # best-effort


def to_static_path(storage_path: str) -> str:
    """Convert DB storage_path to a safe relative path for /static URLs."""
    root = get_image_storage_root()
    root_str = str(root).strip("/")
    sp = storage_path.lstrip("/")

    # If already rooted-relative like 'data_store/<uuid>.jpg', strip prefix.
    if root_str and sp.startswith(root_str + "/"):
        sp = sp[len(root_str) + 1 :]

    # If absolute, attempt to relativize to root; otherwise fall back to name.
    try:
        p = Path(storage_path)
        if p.is_absolute():
            try:
                sp = str(p.relative_to(root))
            except Exception:
                sp = p.name
    except Exception:
        pass

    return sp
