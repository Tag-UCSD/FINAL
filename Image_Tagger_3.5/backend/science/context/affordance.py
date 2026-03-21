"""
Affordance Prediction Analyzer — Environmental Activity Suitability Scoring.

Predicts how suitable an indoor scene is for 5 activities (affordances):
  L059  Sleep (Primary)          — Rest & Recovery
  L079  Cook (Daily)             — Food & Drink
  L091  Computer Work (Solo)     — Focused Knowledge Work
  L130  Casual Conversation      — Leisure & Entertainment
  L141  Yoga / Stretching        — Movement & Fitness

Architecture:
  1. Reuses OneFormer ADE20K segmentation from frame.metadata (L1.5 cache).
  2. Maps ADE20K class IDs → COCO-133 panoptic labels (the feature space
     the models were trained on).
  3. Extracts a 310-dim raw feature vector (object presence/counts, pairwise
     2D distances, room-level aggregates) matching the research pipeline.
  4. Runs per-affordance LightGBM regressors to predict suitability scores
     on a 1–7 Likert scale.

The LightGBM models are trained offline from a Hypersim-based pilot dataset
annotated by a Qwen2-VL vision-language model.  See:
  backend/scripts/train_affordance_models.py

References:
  project/src/features/extract_features.py   — original feature extraction
  project/src/models/train_lgbm.py           — Model B training (raw features)
"""

import json
import logging
import math
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.science.core import AnalysisFrame

logger = logging.getLogger("v3.science.affordance")

# ── Paths ────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parent / "data" / "affordance_models"

AFFORDANCE_IDS = ["L059", "L079", "L091", "L130", "L141"]

AFFORDANCE_NAMES = {
    "L059": "Sleep (Primary)",
    "L079": "Cook (Daily)",
    "L091": "Computer Work (Solo)",
    "L130": "Casual Conversation",
    "L141": "Yoga / Stretching",
}

# ── ADE20K → COCO-133 Class Mapping ─────────────────────────────────────────
# OneFormer ADE20K uses 150 semantic classes. The affordance models were
# trained on COCO-133 panoptic features. This mapping bridges the two.
# Keys are ADE20K class names (lowercase); values are COCO-133 labels.
# Only classes with a reasonable COCO equivalent are listed.

ADE20K_TO_COCO: Dict[str, str] = {
    # Furniture / seating
    "bed": "bed",
    "chair": "chair",
    "armchair": "chair",
    "swivel chair": "chair",
    "sofa": "couch",
    "table": "dining table",
    "desk": "dining table",
    "coffee table": "dining table",
    "pool table": "dining table",
    "bench": "bench",
    "stool": "chair",
    "seat": "chair",
    # Electronics
    "television receiver": "tv",
    "tv": "tv",
    "monitor": "tv",
    "screen": "tv",
    "computer": "laptop",
    "laptop": "laptop",
    "crt screen": "tv",
    # Kitchen
    "stove": "oven",
    "oven": "oven",
    "microwave": "microwave",
    "refrigerator": "refrigerator",
    "dishwasher": "refrigerator",
    "sink": "sink",
    "countertop": "counter",
    "counter": "counter",
    # Bathroom
    "toilet": "toilet",
    "bathtub": "toilet",
    # Lighting
    "lamp": "light",
    "chandelier": "light",
    "light": "light",
    "sconce": "light",
    # Storage / shelving
    "bookcase": "shelf",
    "shelf": "shelf",
    "cabinet": "furniture-other-merged",
    "chest of drawers": "furniture-other-merged",
    "wardrobe": "furniture-other-merged",
    # Textiles / soft furnishing
    "pillow": "pillow",
    "cushion": "pillow",
    "blanket": "blanket",
    "curtain": "curtain",
    "blind": "window-blind",
    # Surfaces / structure
    "floor": "floor-other-merged",
    "wall": "wall-other",
    "ceiling": "ceiling-merged",
    "door": "door-stuff",
    "window": "window-other",
    "mirror": "mirror-stuff",
    "stairs": "stairs",
    "column": "structural-other-merged",
    # Decor / misc
    "plant": "potted plant",
    "flower": "flower",
    "vase": "vase",
    "painting": "banner",
    "poster": "banner",
    "clock": "clock",
    "book": "book",
    "bottle": "bottle",
    "glass": "wine glass",
    "cup": "cup",
    "bowl": "bowl",
    "plate": "bowl",
    "towel": "towel",
    "rug": "textile-other-merged",
    "carpet": "textile-other-merged",
    "person": "person",
}

# ── COCO-133 Canonical Class Order ───────────────────────────────────────────
# Must match the order used during training.

COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard",
    "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit",
    "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform",
    "playingfield", "railroad", "river", "road", "roof", "sand", "sea",
    "shelf", "snow", "stairs", "tent", "towel", "tree", "tree-merged",
    "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone",
    "wall-tile", "wall-wood", "water-other", "window-blind", "window-other",
    "ceiling-merged", "door-stuff-merged", "floor-other-merged",
    "food-other-merged", "furniture-other-merged", "ground-other-merged",
    "mountain-merged", "plant-other-merged", "sky-other-merged",
    "structural-other-merged", "textile-other-merged", "vegetation",
]

# ── Pairwise Specification ───────────────────────────────────────────────────
# Conceptual pair aliases → actual COCO class labels

PAIR_ALIASES: Dict[str, str] = {
    "stove": "oven",
    "nightstand": "table-merged",
    "lamp": "light",
    "desk": "table-merged",
    "monitor": "tv",
    "sofa": "couch",
    "coffee_table": "table-merged",
    "wall_proxy": "wall-other-merged",
}

# Note: "table-merged" doesn't exist in COCO-133 but was used in the
# original pipeline for merged table classes. We map it to "dining table"
# at resolution time when building features, since that's the actual COCO label.
_PAIR_RESOLVE: Dict[str, str] = {
    "table-merged": "dining table",
    "wall-other-merged": "wall-other",
}

RAW_PAIRS: List[Tuple[str, str, str]] = [
    # L079 Cook
    ("stove", "counter", "L079"),
    ("stove", "sink", "L079"),
    ("sink", "counter", "L079"),
    ("stove", "refrigerator", "L079"),
    ("counter", "refrigerator", "L079"),
    # L059 Sleep
    ("bed", "nightstand", "L059"),
    ("bed", "lamp", "L059"),
    ("bed", "wall_proxy", "L059"),
    # L091 Computer Work
    ("desk", "chair", "L091"),
    ("desk", "monitor", "L091"),
    ("chair", "monitor", "L091"),
    # L130 Conversation
    ("sofa", "sofa", "L130"),
    ("sofa", "coffee_table", "L130"),
    ("chair", "chair", "L130"),
    ("sofa", "chair", "L130"),
]

FURNITURE_LABELS = {
    "chair", "dining table", "couch", "bed",
    "furniture-other-merged", "shelf",
}

IMAGE_DIAGONAL = math.sqrt(2.0)


# ── Helper Functions ─────────────────────────────────────────────────────────

def _safe(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_").replace("/", "_")


def _resolve_alias(name: str) -> str:
    coco = PAIR_ALIASES.get(name, name)
    return _PAIR_RESOLVE.get(coco, coco)


def _col_dist(n1: str, n2: str) -> str:
    return f"dist_{_safe(n1)}_{_safe(n2)}"


def _col_ddiff(n1: str, n2: str) -> str:
    return f"depth_diff_{_safe(n1)}_{_safe(n2)}"


# ── Segment Conversion ───────────────────────────────────────────────────────

def _detections_to_segments(
    detections,
    id2label: Dict[int, str],
    image_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """
    Convert OneFormer sv.Detections → list of segment dicts compatible with
    the affordance feature extraction pipeline.

    Each segment dict contains:
      coco_class_label, segment_id, centroid, area_fraction, is_thing
    """
    H, W = image_shape[:2]
    total_pixels = H * W
    segments = []

    # COCO thing class labels (first 80 are "things" in COCO panoptic)
    thing_labels = set(COCO_CLASSES[:80])

    for i, (mask, cls_id, conf) in enumerate(
        zip(detections.mask, detections.class_id, detections.confidence)
    ):
        ade_label = id2label.get(int(cls_id), f"class_{cls_id}").lower()
        coco_label = ADE20K_TO_COCO.get(ade_label)
        if coco_label is None:
            # Try partial match
            for ade_key, coco_val in ADE20K_TO_COCO.items():
                if ade_key in ade_label or ade_label in ade_key:
                    coco_label = coco_val
                    break
        if coco_label is None:
            continue  # No COCO equivalent; skip

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        cx = float(xs.mean()) / W
        cy = float(ys.mean()) / H
        area_fraction = float(mask.sum()) / total_pixels

        segments.append({
            "segment_id": i + 1,
            "coco_class_label": coco_label,
            "centroid": [cx, cy],
            "area_fraction": area_fraction,
            "is_thing": coco_label in thing_labels,
        })

    return segments


# ── Feature Extraction ───────────────────────────────────────────────────────

def compute_presence_counts(
    segments: List[Dict], all_labels: List[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    raw_counts: Dict[str, int] = defaultdict(int)
    for seg in segments:
        raw_counts[seg["coco_class_label"]] += 1

    presence = {}
    count = {}
    for label in all_labels:
        n = raw_counts.get(label, 0)
        presence[f"presence_{_safe(label)}"] = 1 if n > 0 else 0
        count[f"count_{_safe(label)}"] = n
    return presence, count


def _euclidean_2d(c1: List[float], c2: List[float]) -> float:
    dx, dy = c1[0] - c2[0], c1[1] - c2[1]
    return math.sqrt(dx * dx + dy * dy)


def _find_closest_pair(
    segs_a: List[Dict], segs_b: List[Dict], same_class: bool,
) -> Tuple[float, Optional[int], Optional[int]]:
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


def compute_pairwise_features(
    segments: List[Dict],
) -> Dict[str, float]:
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for seg in segments:
        by_label[seg["coco_class_label"]].append(seg)

    dist_out: Dict[str, float] = {}
    ddiff_out: Dict[str, float] = {}

    for name1, name2, _ in RAW_PAIRS:
        dcol = _col_dist(name1, name2)
        ddcol = _col_ddiff(name1, name2)
        if dcol in dist_out:
            continue

        label1 = _resolve_alias(name1)
        label2 = _resolve_alias(name2)
        same = (label1 == label2)
        segs_a = by_label.get(label1, [])
        segs_b = by_label.get(label2, [])

        min_d, _, _ = _find_closest_pair(segs_a, segs_b, same_class=same)
        dist_out[dcol] = min_d
        # No depth info available from OneFormer; use sentinel
        ddiff_out[ddcol] = -1.0

    out = {}
    out.update(dist_out)
    out.update(ddiff_out)
    return out


def compute_room_aggregates(segments: List[Dict]) -> Dict[str, float]:
    things = [s for s in segments if s.get("is_thing", False)]
    stuffs = [s for s in segments if not s.get("is_thing", True)]

    thing_areas = [s["area_fraction"] for s in things]
    all_centroids_y = [s["centroid"][1] for s in segments]

    total_object_count = len(things)
    total_stuff_count = len(stuffs)
    num_unique_classes = len({s["coco_class_label"] for s in segments})
    free_floor_fraction = max(0.0, 1.0 - sum(thing_areas))
    largest_object_area = max(thing_areas) if thing_areas else 0.0
    furniture_area = sum(
        s["area_fraction"] for s in things
        if s["coco_class_label"] in FURNITURE_LABELS
    )
    mean_object_area = float(np.mean(thing_areas)) if thing_areas else 0.0
    std_object_area = float(np.std(thing_areas, ddof=0)) if len(thing_areas) > 1 else 0.0
    scene_complexity = total_object_count * num_unique_classes
    vertical_spread = (
        float(max(all_centroids_y) - min(all_centroids_y))
        if len(all_centroids_y) >= 2 else 0.0
    )

    return {
        "total_object_count": float(total_object_count),
        "total_stuff_count": float(total_stuff_count),
        "num_unique_classes": float(num_unique_classes),
        "free_floor_fraction": free_floor_fraction,
        "largest_object_area": largest_object_area,
        "furniture_clutter_index": furniture_area,
        "mean_object_area": mean_object_area,
        "std_object_area": std_object_area,
        "scene_complexity": float(scene_complexity),
        "vertical_spread": vertical_spread,
        # No depth data from OneFormer
        "depth_available": 0.0,
        "scene_depth_median_m": 0.0,
        "scene_depth_range_m": 0.0,
        "scene_depth_std_m": 0.0,
    }


def extract_feature_vector(
    segments: List[Dict],
    feature_cols: List[str],
) -> np.ndarray:
    """
    Build the full raw feature vector from segment dicts.
    Returns a 1-D float32 array aligned with feature_cols.
    """
    presence, count = compute_presence_counts(segments, COCO_CLASSES)
    pairwise = compute_pairwise_features(segments)
    aggregates = compute_room_aggregates(segments)

    all_features: Dict[str, float] = {}
    all_features.update(presence)
    all_features.update(count)
    all_features.update(pairwise)
    all_features.update(aggregates)

    vec = np.zeros(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        vec[i] = all_features.get(col, 0.0)
    return vec


# ── Model Loading ────────────────────────────────────────────────────────────

_MODELS: Dict[str, Any] = {}
_FEATURE_COLS: Optional[List[str]] = None


def _load_feature_cols() -> List[str]:
    global _FEATURE_COLS
    if _FEATURE_COLS is not None:
        return _FEATURE_COLS

    path = _DATA_DIR / "feature_columns.json"
    if not path.exists():
        logger.warning("Affordance feature_columns.json not found at %s", path)
        return []

    with open(path) as f:
        _FEATURE_COLS = json.load(f)
    logger.info("Loaded %d affordance feature columns", len(_FEATURE_COLS))
    return _FEATURE_COLS


def _load_model(aff_id: str):
    if aff_id in _MODELS:
        return _MODELS[aff_id]

    path = _DATA_DIR / aff_id / "lgbm_model.pkl"
    if not path.exists():
        logger.warning("Affordance model not found: %s", path)
        _MODELS[aff_id] = None
        return None

    with open(path, "rb") as f:
        model = pickle.load(f)
    _MODELS[aff_id] = model
    logger.info("Loaded affordance model for %s", aff_id)
    return model


# ── Analyzer ─────────────────────────────────────────────────────────────────

class AffordanceAnalyzer:
    """
    Predicts environmental affordance scores from OneFormer segmentation.

    Requires enable_segmentation=True in the pipeline config (L1.5 must run
    first so that detections are cached in frame.metadata).

    Output attributes:
      affordance.L059  (1.0–7.0)  Sleep suitability
      affordance.L079  (1.0–7.0)  Cook suitability
      affordance.L091  (1.0–7.0)  Computer Work suitability
      affordance.L130  (1.0–7.0)  Conversation suitability
      affordance.L141  (1.0–7.0)  Yoga suitability
      affordance.L059_norm (0.0–1.0) Normalized score
      ... etc.
    """

    def __init__(self):
        self._initialized = False

    def _ensure_init(self) -> bool:
        if self._initialized:
            return bool(_FEATURE_COLS)
        self._initialized = True
        cols = _load_feature_cols()
        if not cols:
            return False
        for aff_id in AFFORDANCE_IDS:
            _load_model(aff_id)
        return True

    def analyze(self, frame: AnalysisFrame) -> None:
        """
        Run affordance prediction on the given frame.

        Expects OneFormer segmentation results in frame.metadata
        (set by SegmentationAnalyzer).
        """
        try:
            if not self._ensure_init():
                logger.warning(
                    "AffordanceAnalyzer: models not available. "
                    "Run backend/scripts/train_affordance_models.py first."
                )
                return

            detections = frame.metadata.get("oneformer_detections")
            if detections is None or len(detections) == 0:
                logger.warning(
                    "AffordanceAnalyzer: no segmentation detections in frame. "
                    "Enable segmentation (L1.5) before affordance analysis."
                )
                return

            visualizer = frame.metadata.get("oneformer_visualizer")
            if visualizer is None:
                logger.warning("AffordanceAnalyzer: no visualizer cached; cannot resolve labels.")
                return

            id2label = visualizer.id2label
            image_shape = frame.original_image.shape[:2]

            # Convert OneFormer detections → COCO segment dicts
            segments = _detections_to_segments(detections, id2label, image_shape)

            if not segments:
                logger.info("AffordanceAnalyzer: no COCO-mappable segments found.")
                return

            # Extract feature vector
            feature_cols = _load_feature_cols()
            X = extract_feature_vector(segments, feature_cols)

            # Predict each affordance
            for aff_id in AFFORDANCE_IDS:
                model = _load_model(aff_id)
                if model is None:
                    continue

                score = float(model.predict(X.reshape(1, -1))[0])
                # Clamp to valid Likert range
                score = max(1.0, min(7.0, score))
                normalized = (score - 1.0) / 6.0

                frame.add_attribute(f"affordance.{aff_id}", score, confidence=0.85)
                frame.add_attribute(f"affordance.{aff_id}_norm", normalized, confidence=0.85)

            # Store provenance
            frame.metadata["affordance.n_segments"] = len(segments)
            frame.metadata["affordance.segment_classes"] = list(
                {s["coco_class_label"] for s in segments}
            )

            logger.info(
                "AffordanceAnalyzer: predicted %d affordances from %d segments",
                len(AFFORDANCE_IDS), len(segments),
            )

        except Exception as exc:
            logger.error("AffordanceAnalyzer failed: %s", exc, exc_info=True)


# ── Standalone utility ───────────────────────────────────────────────────────

def predict_affordances_from_image(
    image: np.ndarray,
) -> Dict[str, float]:
    """
    Convenience function: run segmentation + affordance prediction on a raw
    RGB image. Returns {affordance_id: score} dict.

    Useful for testing outside the full pipeline.
    """
    frame = AnalysisFrame(image_id=-1, original_image=image)

    from backend.science.vision.segmentation import SegmentationAnalyzer
    SegmentationAnalyzer.analyze(frame, use_semantic=True, use_panoptic=True)

    analyzer = AffordanceAnalyzer()
    analyzer.analyze(frame)

    return {
        aff_id: frame.attributes.get(f"affordance.{aff_id}", 0.0)
        for aff_id in AFFORDANCE_IDS
    }
