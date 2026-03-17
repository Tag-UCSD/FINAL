#!/usr/bin/env python3
"""
Extract a structured numeric feature matrix from Mask2Former segmentation outputs.

Five feature groups per image:
  1. Object presence vector       (~133 dims)  presence_{class}
  2. Object count vector          (~133 dims)  count_{class}
  3. Key pairwise 2D distances    (~15 dims)   dist_{obj1}_{obj2}
  4. Key pairwise depth diffs     (~15 dims)   depth_diff_{obj1}_{obj2}  [metres, GT depth]
  5. Room-level aggregates        (~13 dims)   includes 3 depth stats + depth_available flag

Depth features require Hypersim GT distance_from_camera HDF5 files.  Run
scripts/download_hypersim_depth.py first.  If depth files are absent, groups 4
and the depth aggregate columns are filled with the -1 / 0 sentinels and the
script continues without error.

Output:
  project/data/assembled_dataset/features_raw.parquet
  project/data/assembled_dataset/feature_names.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DEFAULT_SEG_DIR  = PROJECT_ROOT / "data" / "segmentation_outputs"
DEFAULT_DEPTH_DIR = PROJECT_ROOT / "data" / "hypersim_pilot_420"
DEFAULT_OUT_DIR  = PROJECT_ROOT / "data" / "assembled_dataset"
DEFAULT_TAX_MAP  = PROJECT_ROOT / "configs" / "coco_to_taxonomy_map.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "hypersim_image_manifest.csv"

# Maximum plausible indoor depth in metres; values beyond this are treated as invalid.
MAX_DEPTH_M = 50.0

# ---------------------------------------------------------------------------
# COCO class label → safe column-name suffix
# ---------------------------------------------------------------------------
def _safe(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Aliases: conceptual pair names → actual COCO class labels in the data
# ---------------------------------------------------------------------------
PAIR_ALIASES: dict[str, str] = {
    "stove":        "oven",
    "nightstand":   "table-merged",
    "lamp":         "light",
    "desk":         "table-merged",
    "monitor":      "tv",
    "sofa":         "couch",
    "coffee_table": "table-merged",
    "wall_proxy":   "wall-other-merged",
}

# ---------------------------------------------------------------------------
# Pairwise specification — (conceptual_name_1, conceptual_name_2, affordance)
# ---------------------------------------------------------------------------
RAW_PAIRS: list[tuple[str, str, str]] = [
    # L079 Cook
    ("stove",   "counter",       "L079"),
    ("stove",   "sink",          "L079"),
    ("sink",    "counter",       "L079"),
    ("stove",   "refrigerator",  "L079"),
    ("counter", "refrigerator",  "L079"),
    # L059 Sleep
    ("bed",     "nightstand",    "L059"),
    ("bed",     "lamp",          "L059"),
    ("bed",     "wall_proxy",    "L059"),
    # L091 Computer Work
    ("desk",    "chair",         "L091"),
    ("desk",    "monitor",       "L091"),
    ("chair",   "monitor",       "L091"),
    # L130 Conversation
    ("sofa",    "sofa",          "L130"),   # same-class pair
    ("sofa",    "coffee_table",  "L130"),
    ("chair",   "chair",         "L130"),   # same-class pair
    ("sofa",    "chair",         "L130"),
    # L141 — no specific pairs (handled by room-level features)
]

FURNITURE_LABELS: set[str] = {
    "chair", "dining table", "table-merged", "couch", "bed",
    "cabinet-merged", "shelf",
}

IMAGE_DIAGONAL = math.sqrt(2.0)  # normaliser for [0,1]×[0,1] centroid space


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _resolve(name: str) -> str:
    return PAIR_ALIASES.get(name, name)


def _col_dist(n1: str, n2: str) -> str:
    return f"dist_{_safe(n1)}_{_safe(n2)}"


def _col_ddiff(n1: str, n2: str) -> str:
    return f"depth_diff_{_safe(n1)}_{_safe(n2)}"


def _euclidean_2d(c1: list[float], c2: list[float]) -> float:
    dx, dy = c1[0] - c2[0], c1[1] - c2[1]
    return math.sqrt(dx * dx + dy * dy)


def _find_closest_pair(
    segs_a: list[dict], segs_b: list[dict], same_class: bool
) -> tuple[float, int | None, int | None]:
    """
    Find the minimum normalised 2D centroid distance across all instance pairs.

    Returns (min_dist, seg_id_a, seg_id_b).
    Returns (-1.0, None, None) if there are insufficient instances.
    For same-class pairs, at least 2 distinct instances are required.
    """
    if same_class:
        if len(segs_a) < 2:
            return -1.0, None, None
        candidate_pairs = list(combinations(segs_a, 2))
    else:
        if not segs_a or not segs_b:
            return -1.0, None, None
        candidate_pairs = [(a, b) for a in segs_a for b in segs_b]

    min_d = float("inf")
    best_a = best_b = None
    for s1, s2 in candidate_pairs:
        d = _euclidean_2d(s1["centroid"], s2["centroid"]) / IMAGE_DIAGONAL
        if d < min_d:
            min_d, best_a, best_b = d, s1["segment_id"], s2["segment_id"]

    return (min_d, best_a, best_b) if min_d < float("inf") else (-1.0, None, None)


# ---------------------------------------------------------------------------
# Depth loading
# ---------------------------------------------------------------------------

def _depth_hdf5_path(row: pd.Series, depth_dir: Path) -> Path:
    """
    Derive the GT depth HDF5 path from a manifest row.

    Manifest file_path:
        scenes/{scene}/images/scene_{cam}_final_preview/frame.XXXX.tonemap.jpg
    Depth HDF5 path (mirroring the zip layout under scenes/):
        {depth_dir}/scenes/{scene}/images/scene_{cam}_geometry_hdf5/
            frame.XXXX.depth_meters.hdf5
    """
    fp = row["file_path"]
    depth_rel = (
        fp.replace("_final_preview/frame.", "_geometry_hdf5/frame.")
          .replace(".tonemap.jpg", ".depth_meters.hdf5")
    )
    return depth_dir / depth_rel


def load_depth_map(depth_path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """
    Load a Hypersim distance_from_camera HDF5 file and return a float32 array.

    Invalid pixels (inf, NaN, ≤ 0, > MAX_DEPTH_M) are set to NaN.
    If target_shape (H, W) is given and differs from the file shape, bilinear
    resize is applied so the depth map aligns with the panoptic_map.
    Returns None on any error.
    """
    if not _H5PY_AVAILABLE:
        return None
    if not depth_path.exists():
        return None
    try:
        with h5py.File(depth_path, "r") as fh:
            depth = np.array(fh["dataset"], dtype=np.float32)
    except Exception as exc:
        logging.warning("Could not load depth %s: %s", depth_path.name, exc)
        return None

    # Mask invalid values
    depth = np.where(np.isfinite(depth) & (depth > 0) & (depth < MAX_DEPTH_M), depth, np.nan)

    # Resize to match panoptic_map if needed
    if target_shape is not None and depth.shape != target_shape:
        from PIL import Image
        valid_mask = np.isfinite(depth)
        # Replace NaN with 0 for PIL, then restore after resize
        depth_fill = np.where(valid_mask, depth, 0.0).astype(np.float32)
        d_img = Image.fromarray(depth_fill)
        d_img = d_img.resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR)
        depth = np.array(d_img, dtype=np.float32)
        # Zero-filled borders will have near-zero values; treat 0 as invalid
        depth = np.where(depth > 0, depth, np.nan)

    return depth


# ---------------------------------------------------------------------------
# Per-segment depth computation
# ---------------------------------------------------------------------------

def compute_segment_depths(
    segments: list[dict],
    panoptic_map: np.ndarray,
    depth_map: np.ndarray,
) -> dict[int, float]:
    """
    For each segment, compute the median of valid depth pixels within its mask.

    Returns {segment_id: median_depth_m}.  Segments with no valid depth pixels
    are excluded from the dict (treated as unknown depth).
    """
    seg_depths: dict[int, float] = {}
    for seg in segments:
        sid = seg["segment_id"]
        mask = panoptic_map == sid
        depth_vals = depth_map[mask]
        valid = depth_vals[np.isfinite(depth_vals)]
        if valid.size > 0:
            seg_depths[sid] = float(np.median(valid))
    return seg_depths


def compute_scene_depth_stats(depth_map: np.ndarray) -> dict[str, float]:
    """Whole-image depth statistics over valid pixels."""
    valid = depth_map[np.isfinite(depth_map)]
    if valid.size == 0:
        return {"scene_depth_median_m": 0.0, "scene_depth_range_m": 0.0, "scene_depth_std_m": 0.0}
    return {
        "scene_depth_median_m": float(np.median(valid)),
        "scene_depth_range_m":  float(valid.max() - valid.min()),
        "scene_depth_std_m":    float(np.std(valid)),
    }


# ---------------------------------------------------------------------------
# Feature Group 1 & 2: Presence and counts
# ---------------------------------------------------------------------------

def compute_presence_counts(
    segments: list[dict], all_labels: list[str]
) -> tuple[dict[str, int], dict[str, int]]:
    raw_counts: dict[str, int] = defaultdict(int)
    for seg in segments:
        raw_counts[seg["coco_class_label"]] += 1

    presence: dict[str, int] = {}
    count: dict[str, int] = {}
    for label in all_labels:
        n = raw_counts.get(label, 0)
        presence[f"presence_{_safe(label)}"] = 1 if n > 0 else 0
        count[f"count_{_safe(label)}"] = n
    return presence, count


# ---------------------------------------------------------------------------
# Feature Groups 3 & 4: Pairwise 2D distances + depth diffs
# ---------------------------------------------------------------------------

def compute_pairwise_features(
    segments: list[dict],
    seg_depths: dict[int, float] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Returns (dist_features, depth_diff_features).

    dist_features:       {dist_{n1}_{n2}: min_normalised_2D_dist | -1}
    depth_diff_features: {depth_diff_{n1}_{n2}: |depth_a - depth_b| in metres | -1}

    depth_diff is -1 if objects absent OR depth not available for either instance.
    Both features use the same minimum-2D-distance instance pair for consistency.
    """
    by_label: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        by_label[seg["coco_class_label"]].append(seg)

    dist_out: dict[str, float] = {}
    ddiff_out: dict[str, float] = {}

    for name1, name2, _ in RAW_PAIRS:
        dcol  = _col_dist(name1, name2)
        ddcol = _col_ddiff(name1, name2)
        if dcol in dist_out:
            continue  # dedup same-column pairs

        label1 = _resolve(name1)
        label2 = _resolve(name2)
        same   = (label1 == label2)
        segs_a = by_label.get(label1, [])
        segs_b = by_label.get(label2, [])

        min_d, sid_a, sid_b = _find_closest_pair(segs_a, segs_b, same_class=same)
        dist_out[dcol] = min_d

        # Depth diff: use the same winning instance pair
        if sid_a is None or sid_b is None or seg_depths is None:
            ddiff_out[ddcol] = -1.0
        else:
            d_a = seg_depths.get(sid_a)
            d_b = seg_depths.get(sid_b)
            if d_a is None or d_b is None:
                ddiff_out[ddcol] = -1.0
            else:
                ddiff_out[ddcol] = abs(d_a - d_b)

    return dist_out, ddiff_out


# ---------------------------------------------------------------------------
# Feature Group 5: Room-level aggregates (extended with depth stats)
# ---------------------------------------------------------------------------

def compute_room_aggregates(
    segments: list[dict],
    depth_map: np.ndarray | None,
) -> dict[str, float]:
    things = [s for s in segments if s.get("is_thing", False)]
    stuffs = [s for s in segments if not s.get("is_thing", True)]

    thing_areas     = [s["area_fraction"] for s in things]
    all_centroids_y = [s["centroid"][1]   for s in segments]

    total_object_count  = len(things)
    total_stuff_count   = len(stuffs)
    num_unique_classes  = len({s["coco_class_label"] for s in segments})
    free_floor_fraction = max(0.0, 1.0 - sum(thing_areas))
    largest_object_area = max(thing_areas) if thing_areas else 0.0
    furniture_area      = sum(
        s["area_fraction"] for s in things
        if s["coco_class_label"] in FURNITURE_LABELS
    )
    mean_object_area = float(np.mean(thing_areas))   if thing_areas          else 0.0
    std_object_area  = float(np.std(thing_areas, ddof=0)) if len(thing_areas) > 1 else 0.0
    scene_complexity = total_object_count * num_unique_classes
    vertical_spread  = (
        float(max(all_centroids_y) - min(all_centroids_y))
        if len(all_centroids_y) >= 2 else 0.0
    )

    agg: dict[str, float] = {
        "total_object_count":      float(total_object_count),
        "total_stuff_count":       float(total_stuff_count),
        "num_unique_classes":      float(num_unique_classes),
        "free_floor_fraction":     free_floor_fraction,
        "largest_object_area":     largest_object_area,
        "furniture_clutter_index": furniture_area,
        "mean_object_area":        mean_object_area,
        "std_object_area":         std_object_area,
        "scene_complexity":        float(scene_complexity),
        "vertical_spread":         vertical_spread,
    }

    # Depth aggregates — appended to this group
    if depth_map is not None:
        agg["depth_available"] = 1.0
        agg.update(compute_scene_depth_stats(depth_map))
    else:
        agg["depth_available"]    = 0.0
        agg["scene_depth_median_m"] = 0.0
        agg["scene_depth_range_m"]  = 0.0
        agg["scene_depth_std_m"]    = 0.0

    return agg


# ---------------------------------------------------------------------------
# Per-image orchestration
# ---------------------------------------------------------------------------

def load_segments(seg_dir: Path, image_id: str) -> list[dict]:
    path = seg_dir / f"{image_id}_segments.json"
    if not path.exists():
        return []
    try:
        with path.open() as fh:
            return json.load(fh).get("segments", [])
    except Exception as exc:
        logging.warning("Could not load %s: %s", path.name, exc)
        return []


def load_panoptic_map(seg_dir: Path, image_id: str) -> np.ndarray | None:
    path = seg_dir / f"{image_id}_panoptic.npz"
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)["panoptic_map"]
    except Exception as exc:
        logging.warning("Could not load panoptic map %s: %s", path.name, exc)
        return None


def featurise_image(
    image_id: str,
    row: pd.Series,
    seg_dir: Path,
    depth_dir: Path | None,
    all_labels: list[str],
) -> dict[str, Any]:
    segments = load_segments(seg_dir, image_id)

    # --- Depth loading -------------------------------------------------------
    depth_map:  np.ndarray | None = None
    seg_depths: dict[int, float] | None = None

    if depth_dir is not None and _H5PY_AVAILABLE:
        panoptic_map = load_panoptic_map(seg_dir, image_id)
        if panoptic_map is not None:
            depth_path = _depth_hdf5_path(row, depth_dir)
            depth_map  = load_depth_map(depth_path, target_shape=panoptic_map.shape)
            if depth_map is not None:
                seg_depths = compute_segment_depths(segments, panoptic_map, depth_map)

    # --- Feature computation -------------------------------------------------
    presence, count  = compute_presence_counts(segments, all_labels)
    dist_feats, ddiff_feats = compute_pairwise_features(segments, seg_depths)
    aggregates       = compute_room_aggregates(segments, depth_map)

    out: dict[str, Any] = {"image_id": image_id}
    out.update(presence)
    out.update(count)
    out.update(dist_feats)
    out.update(ddiff_feats)
    out.update(aggregates)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seg-dir",    type=Path, default=DEFAULT_SEG_DIR)
    p.add_argument("--depth-dir",  type=Path, default=DEFAULT_DEPTH_DIR,
                   help="Root containing scenes/.../geometry_hdf5/ depth files. "
                        "Pass 'none' to disable depth features.")
    p.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tax-map",    type=Path, default=DEFAULT_TAX_MAP)
    p.add_argument("--manifest",   type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--max-images", type=int,  default=None)
    p.add_argument("--log-level",  default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    depth_dir: Path | None = None
    if str(args.depth_dir).lower() != "none":
        depth_dir = args.depth_dir
        if not _H5PY_AVAILABLE:
            logging.warning("h5py not installed — depth features will be disabled.")
            depth_dir = None

    # Load canonical class order
    with args.tax_map.open() as fh:
        all_labels: list[str] = list(json.load(fh)["classes"].keys())
    logging.info("Loaded %d COCO classes.", len(all_labels))

    # Load manifest
    manifest = pd.read_csv(args.manifest)
    if args.max_images is not None:
        manifest = manifest.head(args.max_images)
    logging.info("Processing %d images (depth_dir=%s).", len(manifest), depth_dir)

    # ---- Pre-compute column name lists (for feature_names.json) -------------
    presence_cols = [f"presence_{_safe(l)}" for l in all_labels]
    count_cols    = [f"count_{_safe(l)}"    for l in all_labels]
    dist_cols     = list(dict.fromkeys(_col_dist(n1, n2) for n1, n2, _ in RAW_PAIRS))
    ddiff_cols    = list(dict.fromkeys(_col_ddiff(n1, n2) for n1, n2, _ in RAW_PAIRS))
    agg_cols      = [
        "total_object_count", "total_stuff_count", "num_unique_classes",
        "free_floor_fraction", "largest_object_area", "furniture_clutter_index",
        "mean_object_area", "std_object_area", "scene_complexity", "vertical_spread",
        "depth_available", "scene_depth_median_m", "scene_depth_range_m", "scene_depth_std_m",
    ]
    feature_names = {
        "presence":      presence_cols,
        "count":         count_cols,
        "pairwise_2d":   dist_cols,
        "pairwise_depth": ddiff_cols,
        "aggregates":    agg_cols,
    }

    # ---- Process images ------------------------------------------------------
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        r = featurise_image(
            image_id  = row.image_id,
            row       = pd.Series(row._asdict()),
            seg_dir   = args.seg_dir,
            depth_dir = depth_dir,
            all_labels= all_labels,
        )
        rows.append(r)

    elapsed = time.perf_counter() - t0
    logging.info(
        "Feature extraction complete in %.1fs (%.3fs/image).",
        elapsed, elapsed / max(len(manifest), 1),
    )

    # ---- Assemble and clean DataFrame ----------------------------------------
    df = pd.DataFrame(rows).set_index("image_id")

    sentinel_cols = dist_cols + ddiff_cols
    other_cols    = [c for c in df.columns if c not in sentinel_cols]
    df[other_cols]    = df[other_cols].fillna(0)
    df[sentinel_cols] = df[sentinel_cols].fillna(-1)

    residual_nan = df.isnull().sum().sum()
    if residual_nan > 0:
        logging.warning("Residual NaN after fill: %d — forcing 0.", residual_nan)
        df = df.fillna(0)

    # ---- Save ----------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / "features_raw.parquet"
    names_path   = args.out_dir / "feature_names.json"

    df.to_parquet(parquet_path)
    with names_path.open("w") as fh:
        json.dump(feature_names, fh, indent=2)

    logging.info("Saved %s", parquet_path)
    logging.info("Saved %s", names_path)

    # ---- Summary -------------------------------------------------------------
    depth_coverage = int(df["depth_available"].sum())
    print("\n" + "=" * 62)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 62)
    print(f"Shape            : {df.shape}  ({df.shape[0]} images × {df.shape[1]} features)")
    print(f"NaN values       : {df.isnull().sum().sum()}")
    print(f"Depth coverage   : {depth_coverage} / {len(df)} images ({100*depth_coverage/max(len(df),1):.1f}%)")

    print("\nFeature group sizes:")
    total_cols = 0
    for grp, cols in feature_names.items():
        print(f"  {grp:18s}: {len(cols):3d}")
        total_cols += len(cols)
    print(f"  {'TOTAL':18s}: {total_cols:3d}")

    print("\nSentinel rates (value == -1):")
    for col_group, label in [(dist_cols, "pairwise_2d"), (ddiff_cols, "pairwise_depth")]:
        n_sentinel = (df[col_group] == -1).sum().sum()
        n_total    = len(col_group) * len(df)
        print(f"  {label:18s}: {n_sentinel}/{n_total} ({100*n_sentinel/max(n_total,1):.1f}%)")

    print("\nTop-5 most common objects (total instance count):")
    totals = df[count_cols].sum().sort_values(ascending=False)
    for col, val in totals.head(5).items():
        print(f"  {col[len('count_'):]:30s}: {int(val)}")

    if depth_coverage > 0:
        print("\nDepth aggregate means (images with depth):")
        depth_rows = df[df["depth_available"] == 1]
        for col in ["scene_depth_median_m", "scene_depth_range_m", "scene_depth_std_m"]:
            print(f"  {col}: {depth_rows[col].mean():.2f} m")
        print("\nDepth diff means (non-sentinel only):")
        for col in ddiff_cols:
            non_sentinel = df[col][df[col] != -1]
            if len(non_sentinel):
                print(f"  {col}: {non_sentinel.mean():.2f} m  (n={len(non_sentinel)})")

    print(f"\nParquet        : {parquet_path}")
    print(f"Feature names  : {names_path}")
    print("=" * 62)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
