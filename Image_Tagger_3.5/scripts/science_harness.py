"""Standalone science-only harness.

Usage (folder mode):

  python -m scripts.science_harness --input-dir path/to/images --output science_attributes.csv

This does not require a running database; it loads images directly from a folder,
runs the science analyzers, and writes a CSV with one row per image.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from backend.services.vlm import describe_vlm_configuration
from typing import List

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from backend.science.core import AnalysisFrame
from backend.science.math.color import ColorAnalyzer
from backend.science.math.complexity import ComplexityAnalyzer
from backend.science.math.glcm import TextureAnalyzer
from backend.science.math.fractals import FractalAnalyzer
from backend.science.spatial.depth import DepthAnalyzer
from backend.science.context.cognitive import CognitiveStateAnalyzer


def iter_images(input_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    files = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Science-only harness for Image Tagger v3.3")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder of images to analyze")
    parser.add_argument("--output", type=str, default="science_attributes.csv", help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if cv2 is None:
        raise SystemExit("OpenCV (cv2) is required to run the science harness.")

    images = iter_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    color = ColorAnalyzer()
    comp = ComplexityAnalyzer()
    tex = TextureAnalyzer()
    frac = FractalAnalyzer()
    depth = DepthAnalyzer()
    cognitive = CognitiveStateAnalyzer()  # neutral baseline

    rows = []
    all_keys = set()

    for idx, path in enumerate(images):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        frame = AnalysisFrame(image_id=idx, original_image=rgb)

        color.analyze(frame)
        comp.analyze(frame)
        tex.analyze(frame)
        frac.analyze(frame)
        depth.analyze(frame)
        cognitive.analyze(frame)

        row = {"filename": path.name}
        row.update(frame.attributes)
        rows.append(row)
        all_keys.update(frame.attributes.keys())

    all_keys = sorted(all_keys)
    fieldnames = ["filename"] + all_keys

    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()