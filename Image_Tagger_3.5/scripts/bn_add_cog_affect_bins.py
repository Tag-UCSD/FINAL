"""BN helper: add discrete cognitive/affective bins to a BN snapshot CSV.

Usage:
    python scripts/bn_add_cog_affect_bins.py

Assumes:
    - Input snapshot at data/bn_snapshot_raw.csv
    - Writes output to data/bn_snapshot_with_bins.csv

The script looks for continuous columns such as:
    cognitive.coherence, cognitive.complexity, cognitive.legibility,
    cognitive.mystery, cognitive.restoration,
    affect.cozy, affect.welcoming, affect.tranquil,
    affect.scary, affect.jarring

and adds 3-state discretized columns:
    COHERENCE, COMPLEXITY, LEGIBILITY, MYSTERY, RESTORATION,
    AFFECT_COSY, AFFECT_WELCOMING, AFFECT_TRANQUIL,
    AFFECT_SCARY, AFFECT_JARRING
"""

import pandas as pd
from pathlib import Path

SNAPSHOT_IN = Path("data/bn_snapshot_raw.csv")
SNAPSHOT_OUT = Path("data/bn_snapshot_with_bins.csv")

def bin_3(x):
    """Map a scalar in [0, 1] to LOW/MID/HIGH.

    Values outside [0, 1] or non-numerical entries are mapped to MID
    as a neutral fallback.
    """
    try:
        v = float(x)
    except Exception:
        return "MID"
    if v < 0.33:
        return "LOW"
    if v < 0.66:
        return "MID"
    return "HIGH"

mapping = {
    "cognitive.coherence": "COHERENCE",
    "cognitive.complexity": "COMPLEXITY",
    "cognitive.legibility": "LEGIBILITY",
    "cognitive.mystery": "MYSTERY",
    "cognitive.restoration": "RESTORATION",
    "affect.cozy": "AFFECT_COSY",
    "affect.welcoming": "AFFECT_WELCOMING",
    "affect.tranquil": "AFFECT_TRANQUIL",
    "affect.scary": "AFFECT_SCARY",
    "affect.jarring": "AFFECT_JARRING",
}

def main():
    if not SNAPSHOT_IN.exists():
        raise SystemExit(f"[bn_add_cog_affect_bins] Input snapshot not found: {SNAPSHOT_IN}")
    df = pd.read_csv(SNAPSHOT_IN)

    for col, node in mapping.items():
        if col in df.columns:
            df[node] = df[col].apply(bin_3)
        else:
            print(f"[bn_add_cog_affect_bins] WARNING: column {col} not found; skipping")

    SNAPSHOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SNAPSHOT_OUT, index=False)
    print(f"[bn_add_cog_affect_bins] wrote {SNAPSHOT_OUT}")

if __name__ == "__main__":
    main()
