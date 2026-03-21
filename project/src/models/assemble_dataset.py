#!/usr/bin/env python3
"""
Assemble the pilot ML dataset: merge features, VLM scores, and indicator features.
Produces pilot_dataset.parquet with train/val/test split at scene level.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "project" / "data"
CONFIGS = ROOT / "project" / "configs"
OUT = DATA / "assembled_dataset"

FEATURES_PATH = OUT / "features_raw.parquet"
MANIFEST_PATH = CONFIGS / "hypersim_image_manifest.csv"
VLM_RAW_DIR = DATA / "vlm_annotations" / "raw"
INDICATOR_VOCAB_PATH = DATA / "vlm_annotations" / "indicator_vocabulary.json"
FEATURE_NAMES_PATH = OUT / "feature_names.json"

OUTPUT_DATASET = OUT / "pilot_dataset.parquet"
OUTPUT_SPLIT = OUT / "split_summary.json"

AFFORDANCES = ["L059", "L079", "L091", "L130", "L141"]
SEED = 42
MIN_INDICATOR_COUNT = 3  # canonical indicators appearing fewer times are excluded


def safe_col(name: str) -> str:
    """Convert indicator name to a safe column name."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())[:60]


def load_canonical_indicators(vocab: dict) -> dict[str, list[dict]]:
    """Return {affordance_id: [{"safe_name": ..., "canonical_polarity": ...}, ...]}"""
    result = {}
    for aff_id in AFFORDANCES:
        entries = vocab.get(aff_id, [])
        seen = set()
        indicators = []
        for entry in entries:
            if entry.get("count", 0) < MIN_INDICATOR_COUNT:
                continue
            sn = safe_col(entry["name"])
            if sn in seen:
                continue
            seen.add(sn)
            indicators.append({
                "safe_name": sn,
                "original_name": entry["name"],
                "canonical_polarity": entry.get("canonical_polarity", "positive"),
            })
        result[aff_id] = indicators
    return result


def build_vlm_scores_df(manifest: pd.DataFrame) -> pd.DataFrame:
    """Read all raw VLM JSONs and build a flat DataFrame with scores."""
    records = []
    for _, row in manifest.iterrows():
        image_id = row["image_id"]
        for aff_id in AFFORDANCES:
            json_path = VLM_RAW_DIR / f"{image_id}_{aff_id}.json"
            if not json_path.exists():
                print(f"  WARNING: missing {json_path.name}")
                continue
            with open(json_path) as f:
                data = json.load(f)
            records.append({
                "image_id": image_id,
                "affordance_id": aff_id,
                "vlm_score": float(data["score"]),
                "vlm_confidence": float(data.get("confidence", 1.0)),
            })
    return pd.DataFrame(records)


def build_indicator_features(
    manifest: pd.DataFrame,
    canonical: dict[str, list[dict]],
) -> pd.DataFrame:
    """
    For each (image_id, affordance_id) build binary ind_pos / ind_neg columns
    based on whether each canonical indicator appears in the raw VLM JSON.
    """
    # Build column names: all ind_pos/ind_neg across all affordances, prefixed with affordance
    all_cols: list[str] = []
    col_meta: list[tuple[str, str, str]] = []  # (col_name, aff_id, polarity)
    for aff_id in AFFORDANCES:
        for ind in canonical[aff_id]:
            sn = ind["safe_name"]
            pos_col = f"ind_{aff_id}_pos_{sn}"
            neg_col = f"ind_{aff_id}_neg_{sn}"
            all_cols.extend([pos_col, neg_col])
            col_meta.extend([
                (pos_col, aff_id, "positive"),
                (neg_col, aff_id, "negative"),
            ])

    print(f"  Total indicator columns: {len(all_cols)}")

    records = []
    for _, row in manifest.iterrows():
        image_id = row["image_id"]
        for aff_id in AFFORDANCES:
            json_path = VLM_RAW_DIR / f"{image_id}_{aff_id}.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                data = json.load(f)

            # Map raw indicator names → {safe_name: polarity}
            raw_indicators: dict[str, str] = {}
            for ind in data.get("indicators", []):
                sn = safe_col(ind["name"])
                raw_indicators[sn] = ind.get("polarity", "positive")

            record = {"image_id": image_id, "affordance_id": aff_id}
            # Only populate columns for this affordance
            for ind in canonical[aff_id]:
                sn = ind["safe_name"]
                polarity_found = raw_indicators.get(sn)
                pos_col = f"ind_{aff_id}_pos_{sn}"
                neg_col = f"ind_{aff_id}_neg_{sn}"
                record[pos_col] = 1 if polarity_found == "positive" else 0
                record[neg_col] = 1 if polarity_found == "negative" else 0

            records.append(record)

    df = pd.DataFrame(records)
    # Fill missing affordance columns with 0
    for col in all_cols:
        if col not in df.columns:
            df[col] = 0
    df[all_cols] = df[all_cols].fillna(0).astype(np.int8)
    return df


def scene_level_split(
    manifest: pd.DataFrame,
) -> tuple[set, set, set]:
    """
    Split at SCENE level, stratified by cluster_assignment.
    Returns (train_scenes, val_scenes, test_scenes).
    """
    # Unique scenes with their cluster
    scenes = (
        manifest[["scene_name", "cluster_assignment"]]
        .drop_duplicates("scene_name")
        .reset_index(drop=True)
    )

    # First split: 80% train+val vs 20% test
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_val_idx, test_idx = next(
        sss_outer.split(scenes, scenes["cluster_assignment"])
    )
    scenes_train_val = scenes.iloc[train_val_idx]
    scenes_test = scenes.iloc[test_idx]

    # Second split: 70/10 → within 80% block, val = 10/80 = 12.5%
    val_frac = 0.10 / 0.80
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=SEED)
    train_idx, val_idx = next(
        sss_inner.split(scenes_train_val, scenes_train_val["cluster_assignment"])
    )
    scenes_train = scenes_train_val.iloc[train_idx]
    scenes_val = scenes_train_val.iloc[val_idx]

    return (
        set(scenes_train["scene_name"]),
        set(scenes_val["scene_name"]),
        set(scenes_test["scene_name"]),
    )


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Assembling pilot dataset")
    print("=" * 60)

    # 1. Load features
    print("\n[1/6] Loading features_raw.parquet...")
    features = pd.read_parquet(FEATURES_PATH)
    features.index.name = "image_id"
    features = features.reset_index()
    print(f"  Features shape: {features.shape}")

    # 2. Load manifest
    print("[2/6] Loading manifest...")
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"  Manifest: {len(manifest)} images, {manifest['cluster_assignment'].nunique()} clusters")

    # 3. Load VLM scores
    print("[3/6] Parsing VLM scores from raw JSONs...")
    vlm_df = build_vlm_scores_df(manifest)
    print(f"  VLM records: {len(vlm_df)}")

    # 4. Load indicator vocabulary
    print("[4/6] Loading indicator vocabulary...")
    with open(INDICATOR_VOCAB_PATH) as f:
        vocab = json.load(f)
    canonical = load_canonical_indicators(vocab)
    total_inds = sum(len(v) for v in canonical.values())
    print(f"  Canonical indicators: {total_inds} across {len(AFFORDANCES)} affordances")
    for aff, inds in canonical.items():
        print(f"    {aff}: {len(inds)} indicators")

    # 5. Build indicator features
    print("[5/6] Building indicator-polarity features...")
    ind_df = build_indicator_features(manifest, canonical)
    print(f"  Indicator feature matrix: {ind_df.shape}")

    # 6. Merge everything
    print("[6/6] Merging and splitting...")

    # Merge features with manifest metadata
    features_with_meta = features.merge(
        manifest[["image_id", "scene_name", "cluster_assignment", "file_path"]],
        on="image_id",
        how="left",
    )

    # Merge VLM scores (cross-join on image_id × affordance_id)
    merged = vlm_df.merge(features_with_meta, on="image_id", how="left")

    # Merge indicator features
    merged = merged.merge(ind_df, on=["image_id", "affordance_id"], how="left")

    print(f"  Merged dataset: {merged.shape}")

    # Scene-level split
    train_scenes, val_scenes, test_scenes = scene_level_split(manifest)

    def assign_split(scene_name: str) -> str:
        if scene_name in train_scenes:
            return "train"
        elif scene_name in val_scenes:
            return "val"
        else:
            return "test"

    merged["split"] = merged["scene_name"].apply(assign_split)

    # Verify split counts
    split_counts = merged.groupby(["split", "affordance_id"]).size().unstack()
    print("\nSplit distribution (rows per affordance):")
    print(split_counts)

    # Save dataset
    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_DATASET, index=False)
    print(f"\nSaved: {OUTPUT_DATASET}")

    # Save split summary
    summary = {
        "total_rows": len(merged),
        "n_images": manifest["image_id"].nunique(),
        "n_affordances": len(AFFORDANCES),
        "affordance_ids": AFFORDANCES,
        "splits": {
            "train": {
                "n_scenes": len(train_scenes),
                "n_rows": int((merged["split"] == "train").sum()),
            },
            "val": {
                "n_scenes": len(val_scenes),
                "n_rows": int((merged["split"] == "val").sum()),
            },
            "test": {
                "n_scenes": len(test_scenes),
                "n_rows": int((merged["split"] == "test").sum()),
            },
        },
        "n_features_raw": len([c for c in merged.columns if c.startswith("presence_") or c.startswith("count_") or c.startswith("dist_") or c.startswith("depth_")]),
        "n_indicator_features": len([c for c in merged.columns if c.startswith("ind_")]),
        "columns": list(merged.columns),
        "cluster_split": {
            split: {
                str(k): int(v)
                for k, v in merged[merged["split"] == split]
                .drop_duplicates("scene_name")["cluster_assignment"]
                .value_counts()
                .items()
            }
            for split in ["train", "val", "test"]
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(OUTPUT_SPLIT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUTPUT_SPLIT}")

    print(f"\nDone in {summary['elapsed_seconds']}s")
    print(f"Dataset: {summary['total_rows']} rows × {len(merged.columns)} columns")
    print(f"  Raw features: {summary['n_features_raw']}")
    print(f"  Indicator features: {summary['n_indicator_features']}")


if __name__ == "__main__":
    main()
