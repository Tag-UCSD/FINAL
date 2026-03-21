"""BN helper: merge cognitive/affective priors into a base priors CSV.

Usage:
    python scripts/bn_merge_cog_affect_priors.py

Assumes:
    - Existing priors CSV at data/BN_PRIORS_BASE.csv
    - Cognitive/affective priors example at docs/BN_PRIORS_COG_AFFECT_EXAMPLE.csv
    - Writes merged priors to data/BN_PRIORS_WITH_COG_AFFECT.csv

The script removes any nodes from the base file that are also present
in the cog/affect priors, then appends the cog/affect rows.
"""

import pandas as pd
from pathlib import Path

BASE_PRIORS = Path("data/BN_PRIORS_BASE.csv")
COG_AFFECT_PRIORS = Path("docs/BN_PRIORS_COG_AFFECT_EXAMPLE.csv")
MERGED_PRIORS = Path("data/BN_PRIORS_WITH_COG_AFFECT.csv")

def main():
    if not BASE_PRIORS.exists():
        raise SystemExit(f"[bn_merge_cog_affect_priors] Base priors not found: {BASE_PRIORS}")
    if not COG_AFFECT_PRIORS.exists():
        raise SystemExit(f"[bn_merge_cog_affect_priors] Cog/affect priors not found: {COG_AFFECT_PRIORS}")

    base = pd.read_csv(BASE_PRIORS)
    extra = pd.read_csv(COG_AFFECT_PRIORS)

    extra_nodes = set(extra["node"].unique())
    base_filtered = base[~base["node"].isin(extra_nodes)]

    merged = pd.concat([base_filtered, extra], ignore_index=True)

    # Sanity check per (node, parents)
    for (node, parents), group in merged.groupby(["node", "parents"]):
        s = group["p"].sum()
        if abs(s - 1.0) > 1e-6:
            print(f"[bn_merge_cog_affect_priors] WARN: probabilities for ({node}, {parents}) sum to {s:.3f}")

    MERGED_PRIORS.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_PRIORS, index=False)
    print(f"[bn_merge_cog_affect_priors] wrote {MERGED_PRIORS}")

if __name__ == "__main__":
    main()
