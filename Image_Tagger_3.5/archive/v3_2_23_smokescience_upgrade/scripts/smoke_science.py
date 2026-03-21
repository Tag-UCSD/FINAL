#!/usr/bin/env python3
"""Lightweight science pipeline smoke test for Image Tagger v3.2.

We import the pipeline and run it on a tiny synthetic image to make
sure the core analysis functions can execute without crashing.
"""

import numpy as np

from backend.science.core import AnalysisFrame
from backend.science.pipeline import run_full_analysis


def main() -> None:
    # Simple 64x64 RGB gradient
    h, w = 64, 64
    y = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1)
    x = np.linspace(0, 255, w, dtype=np.uint8).reshape(1, -1)
    img = np.stack([x.repeat(h, axis=0), y.repeat(w, axis=1), np.full((h, w), 128, dtype=np.uint8)], axis=-1)

    frame = AnalysisFrame(original_image=img, image_id="smoke-test")
    run_full_analysis(frame)
    # Print a small subset of attributes as a sanity check
    sample = dict(list(frame.attributes.items())[:10])
    print("[smoke_science] ran analysis; sample attributes:", sample)


if __name__ == "__main__":
    main()