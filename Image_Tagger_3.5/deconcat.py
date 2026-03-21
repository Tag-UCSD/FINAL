#!/usr/bin/env python3
"""
Deconcatenation helper for Image Tagger multi-file bundles.

This script understands files produced with the following marker format:

    ----- FILE PATH: <relative/path>
    ----- CONTENT START -----
    ... file contents ...
    ----- CONTENT END -----

Usage
-----

    python deconcat.py Image_Tagger_v3.4.63_bn_db_tightening_health_full.txt ./output_dir

The output directory will be created if it does not exist. Existing files
with the same paths will be overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


FILE_PATH_PREFIX = "----- FILE PATH: "
CONTENT_START = "----- CONTENT START -----"
CONTENT_END = "----- CONTENT END -----"


def _iter_lines(fp: TextIO):
    """Yield lines from a file-like object, normalizing newlines."""
    for line in fp:
        # Ensure we work with plain str and strip trailing newlines only
        yield line.rstrip("\n")


def deconcat(concat_path: Path, out_dir: Path) -> None:
    if not concat_path.exists():
        raise FileNotFoundError(concat_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    current_rel: Path | None = None
    buffer: list[str] = []
    in_content = False

    with concat_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for raw_line in _iter_lines(fp):
            if raw_line.startswith(FILE_PATH_PREFIX) and not in_content:
                # Start a new file section
                current_rel = Path(raw_line[len(FILE_PATH_PREFIX) :].strip())
                buffer = []
                continue

            if raw_line == CONTENT_START and current_rel is not None and not in_content:
                in_content = True
                continue

            if raw_line == CONTENT_END and in_content and current_rel is not None:
                # Flush the current buffer to disk
                target_path = out_dir / current_rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text("\n".join(buffer) + "\n", encoding="utf-8")
                # Reset for the next file
                current_rel = None
                buffer = []
                in_content = False
                continue

            if in_content and current_rel is not None:
                buffer.append(raw_line)

    if in_content and current_rel is not None and buffer:
        # Gracefully handle a missing CONTENT_END at EOF by writing
        target_path = out_dir / current_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("\n".join(buffer) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("Usage: python deconcat.py <concatenated.txt> <output-dir>")
        raise SystemExit(1)

    concat_path = Path(args[0]).expanduser().resolve()
    out_dir = Path(args[1]).expanduser().resolve()

    deconcat(concat_path, out_dir)
    print(f"[deconcat] Wrote files into {out_dir}")


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
