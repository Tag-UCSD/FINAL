from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.core import get_db
from backend.services.auth import require_admin
from backend.models.users import User
from backend.models.annotation import Validation
from backend.models.assets import Image
from backend.schemas.supervision import TaggerPerformance, IRRStat, ValidationDetail
from backend.services.storage import to_static_path
from backend.science.index_catalog import get_index_metadata
from backend.api import v1_bn_export


router = APIRouter(prefix="/v1/monitor", tags=["Supervisor Dashboard"])

def _build_restorativeness_heuristic_node(features: List[Dict[str, object]]):
    """Build a simple restorativeness heuristic node for Tag Inspector.

    This is an early, explicitly *heuristic* rule family (H1) that tries to
    combine a few CNfA-aligned visual features into a coarse 3-level judgment
    about how *restorative* an interior might feel.

    Currently we combine:

    - cnfa.biophilic.natural_material_ratio      (more natural material -> more restorative)
    - cnfa.fluency.visual_entropy_spatial        (mid-level visual entropy preferred)
    - cnfa.fluency.clutter_density_count         (less clutter -> more restorative)
    - cnfa.fluency.processing_load_proxy         (lower processing load -> more restorative)

    The computation is intentionally simple and transparent; it should evolve
    as we gather human ratings and fit real models.

    Returns:
        (node, tag) where:
            - node is a BN-like dict for the Tag Inspector `bn.nodes` array
            - tag is a tag dict for the Tag Inspector `tags` array
        Either may be None if we do not have enough evidence.
    """
    # Map feature key -> numeric value for quick lookup.
    feature_values: Dict[str, Optional[float]] = {}
    for row in features:
        key = row.get("key")
        value = row.get("value")
        if not isinstance(key, str):
            continue
        if value is None:
            feature_values[key] = None
            continue
        try:
            feature_values[key] = float(value)
        except (TypeError, ValueError):
            feature_values[key] = None

    nat = feature_values.get("cnfa.biophilic.natural_material_ratio")
    entropy = feature_values.get("cnfa.fluency.visual_entropy_spatial")
    clutter = feature_values.get("cnfa.fluency.clutter_density_count")
    load = feature_values.get("cnfa.fluency.processing_load_proxy")

    evidence = {
        "cnfa.biophilic.natural_material_ratio": nat,
        "cnfa.fluency.visual_entropy_spatial": entropy,
        "cnfa.fluency.clutter_density_count": clutter,
        "cnfa.fluency.processing_load_proxy": load,
    }

    available = {k: v for k, v in evidence.items() if v is not None}
    if len(available) < 2:
        # Not enough data to make even a coarse call.
        return None, None

    def _clamp01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    # Weights are heuristic and explicitly marked as such.
    weights = {}
    if nat is not None:
        weights["nat"] = 0.4
    if entropy is not None:
        weights["entropy"] = 0.3
    if clutter is not None:
        weights["clutter"] = 0.15
    if load is not None:
        weights["load"] = 0.15

    total_w = sum(weights.values())
    if total_w <= 0.0:
        return None, None

    # Natural materials: higher is better.
    nat_term = _clamp01(nat) if nat is not None else 0.5

    # Entropy: we tentatively prefer mid-level values.
    if entropy is not None:
        e = _clamp01(entropy)
        entropy_term = max(0.0, 1.0 - 2.0 * abs(e - 0.5))  # 1 at 0.5, 0 at 0 or 1
    else:
        entropy_term = 0.5

    # Clutter / processing load: lower is better.
    clutter_term = 1.0 - _clamp01(clutter) if clutter is not None else 0.5
    load_term = 1.0 - _clamp01(load) if load is not None else 0.5

    num = 0.0
    num += weights.get("nat", 0.0) * nat_term
    num += weights.get("entropy", 0.0) * entropy_term
    num += weights.get("clutter", 0.0) * clutter_term
    num += weights.get("load", 0.0) * load_term

    rest_score = num / total_w

    # Map score into a coarse 3-bin label.
    if rest_score < 0.33:
        bin_label = "low"
    elif rest_score < 0.66:
        bin_label = "mid"
    else:
        bin_label = "high"

    # Build a short explanation string.
    parts = []
    if nat is not None:
        parts.append(f"natural_material_ratio={nat:.2f}")
    if entropy is not None:
        parts.append(f"visual_entropy_spatial={entropy:.2f}")
    if clutter is not None:
        parts.append(f"clutter_density_count={clutter:.2f}")
    if load is not None:
        parts.append(f"processing_load_proxy={load:.2f}")
    evidence_str = ", ".join(parts) if parts else "no CNfA fluency features available"

    notes = (
        "H1 restorativeness heuristic combining biophilic material ratio and CNfA fluency features. "
        "Higher natural material ratio, mid-level visual entropy, and lower clutter/processing load "
        "push the score toward 'high'. This rule is uncalibrated and should be refined with human "
        "ratings and cultural baselines. Evidence: " + evidence_str
    )

    node_name = "affect.restorative_h1"
    label = "Restorativeness (H1 heuristic)"

    posterior = {bin_label: 1.0} if bin_label else None

    node = {
        "name": node_name,
        "label": label,
        "posterior": posterior,
        "prior": None,
        "notes": notes,
    }

    tag = {
        "key": node_name,
        "label": label,
        "description": notes,
        "value": bin_label,
        "raw_value": rest_score,
        "bin": bin_label,
        "status": "derived",
        "bn_node": node_name,
    }

    return node, tag


@router.get("/velocity", response_model=List[TaggerPerformance])
def get_velocity(
    window_hours: int = 24,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> List[TaggerPerformance]:
    """Aggregate tagger velocity over a recent time window.

    The default window is the last 24 hours. The query aggregates:

      * number of distinct images validated per tagger
      * average dwell time (duration_ms) for those validations

    This endpoint drives the team velocity table in the Supervisor GUI.
    """
    if window_hours <= 0:
        window_hours = 24

    cutoff = datetime.utcnow() - timedelta(hours=window_hours)

    rows = (
        db.query(
            Validation.user_id.label("user_id"),
            func.coalesce(User.username, "unknown").label("username"),
            func.count(func.distinct(Validation.image_id)).label("images_validated"),
            func.coalesce(func.avg(Validation.duration_ms), 0).label("avg_duration_ms"),
            func.count(Validation.id).label("validations_count"),
        )
        .outerjoin(User, User.id == Validation.user_id)
        .filter(Validation.created_at >= cutoff)
        .group_by(Validation.user_id, User.username)
        .all()
    )

    results: List[TaggerPerformance] = []
    for row in rows:
        avg_duration_ms = int(row.avg_duration_ms or 0)
        status = "active" if row.validations_count > 0 else "inactive"
        results.append(
            TaggerPerformance(
                user_id=row.user_id or 0,
                username=row.username or "unknown",
                images_validated=row.images_validated,
                avg_duration_ms=avg_duration_ms,
                status=status,
            )
        )

    return results


@router.get("/irr", response_model=List[IRRStat])
def get_irr(
    window_hours: int = 72,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> List[IRRStat]:
    """Compute a simple IRR metric from overlapping validations.

    For each (image_id, attribute_key) pair with 2+ validations in the
    given time window, we compute:

      * pairwise agreement ratio between raters (exact value match)
      * conflict_count = number of disagreeing pairs

    This is intentionally conservative and easy to interpret, rather than
    a full Fleiss' κ implementation.
    """
    if window_hours <= 0:
        window_hours = 72

    cutoff = datetime.utcnow() - timedelta(hours=window_hours)

    rows = (
        db.query(
            Validation.image_id,
            Validation.attribute_key,
            Validation.value,
            Validation.user_id,
            func.coalesce(User.username, "unknown").label("username"),
            Image.filename,
        )
        .join(Image, Image.id == Validation.image_id)
        .outerjoin(User, User.id == Validation.user_id)
        .filter(Validation.created_at >= cutoff)
        .all()
    )

    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        key = (row.image_id, row.filename, row.attribute_key)
        grouped[key].append((row.value, row.username))

    results: List[IRRStat] = []
    for (image_id, filename, attribute_key), values in grouped.items():
        n = len(values)
        if n < 2:
            continue

        total_pairs = 0
        agree_pairs = 0
        for i in range(n):
            vi, _ui = values[i]
            for j in range(i + 1, n):
                vj, _uj = values[j]
                total_pairs += 1
                if vi == vj:
                    agree_pairs += 1

        if total_pairs == 0:
            continue

        agreement_score = agree_pairs / total_pairs
        conflict_count = total_pairs - agree_pairs
        raters = sorted({u for _v, u in values})

        results.append(
            IRRStat(
                image_id=image_id,
                filename=filename,
                attribute_key=attribute_key,
                agreement_score=agreement_score,
                conflict_count=conflict_count,
                raters=raters,
            )
        )

    return results


@router.get("/image/{image_id}/validations", response_model=List[ValidationDetail])
def get_image_validations(
    image_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> List[ValidationDetail]:
    """Return per-user validations for a specific image.

    This powers the Tag Inspector drawer in the Supervisor dashboard.
    """
    rows = (
        db.query(
            Validation.id,
            Validation.user_id,
            func.coalesce(User.username, "unknown").label("username"),
            Validation.image_id,
            Validation.attribute_key,
            Validation.value,
            Validation.duration_ms,
            Validation.created_at,
        )
        .outerjoin(User, User.id == Validation.user_id)
        .filter(Validation.image_id == image_id)
        .order_by(Validation.created_at.asc())
        .all()
    )

    return [
        ValidationDetail(
            id=row.id,
            user_id=row.user_id,
            username=row.username,
            image_id=row.image_id,
            attribute_key=row.attribute_key,
            value=row.value,
            duration_ms=row.duration_ms,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get("/image/{image_id}/inspector")
def get_image_inspector(
    image_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> Dict[str, object]:
    """Aggregate science pipeline outputs, BN indices, and validations for Tag Inspector.

    This endpoint is intentionally read-only and conservative. It surfaces:

    - basic image metadata and a /static URL,
    - raw science_pipeline_* Validation rows as numeric features,
    - composite indices + bins from the BN export helpers,
    - a simple BN-like node summary for each index,
    - per-user validations for the image (for convenience).
    """

    image = db.query(Image).get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    # Construct a stable /static URL for the underlying image asset.
    storage_path = getattr(image, "storage_path", None) or getattr(image, "filename", "")
    image_url: Optional[str] = None
    if storage_path:
        rel = to_static_path(storage_path)
        image_url = f"/static/{rel}" if rel else None

    # Raw science pipeline attributes (continuous features).
    sci_rows = (
        db.query(Validation)
        .filter(
            Validation.image_id == image_id,
            Validation.source.like("science_pipeline%"),
        )
        .order_by(Validation.attribute_key)
        .all()
    )

    features = [
        {
            "key": row.attribute_key,
            "value": float(row.value) if row.value is not None else None,
            "source": row.source,
        }
        for row in sci_rows
    ]

    # Composite indices and bins reused from the BN export helpers.
    index_meta = get_index_metadata()
    index_keys = list(index_meta.keys())

    indices = v1_bn_export._collect_indices_for_image(db, image_id, index_keys)
    bins = v1_bn_export._collect_bins_for_image(db, image_id)
    irr = v1_bn_export._compute_irr_for_image(db, image_id)

    tags = []
    nodes = []
    for key in index_keys:
        meta = index_meta.get(key, {})
        value = indices.get(key)
        bin_label = bins.get(key)

        if value is None and bin_label is None:
            continue

        tags.append(
            {
                "key": key,
                "label": meta.get("label", key),
                "description": meta.get("description"),
                "value": bin_label if bin_label is not None else value,
                "raw_value": value,
                "bin": bin_label,
                "status": "machine",
                "bn_node": key,
            }
        )

        posterior = {}
        if bin_label:
            posterior[bin_label] = 1.0

        nodes.append(
            {
                "name": key,
                "label": meta.get("label", key),
                "posterior": posterior or None,
                "prior": None,
                "notes": meta.get("description"),
            }
        )

    # H1: add a derived restorativeness heuristic node based on CNfA fluency + biophilia features.
    rest_node, rest_tag = _build_restorativeness_heuristic_node(features)
    if rest_node is not None:
        nodes.append(rest_node)
    if rest_tag is not None:
        tags.append(rest_tag)

    # Also surface per-user validations as part of the inspector payload.
    validations_rows = (
        db.query(
            Validation.id,
            Validation.user_id,
            User.username,
            Validation.image_id,
            Validation.attribute_key,
            Validation.value,
            Validation.duration_ms,
            Validation.created_at,
        )
        .join(User, Validation.user_id == User.id, isouter=True)
        .filter(Validation.image_id == image_id)
        .order_by(Validation.created_at.desc())
        .all()
    )

    validations = [
        {
            "id": row.id,
            "user_id": row.user_id,
            "username": row.username,
            "image_id": row.image_id,
            "attribute_key": row.attribute_key,
            "value": row.value,
            "duration_ms": row.duration_ms,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in validations_rows
    ]

    return {
        "image": {
            "image_id": image.id,
            "filename": image.filename,
            "url": image_url,
        },
        "pipeline": {
            # We do not yet track per-analyzer run state per image; this will evolve.
            "overall_status": "unknown",
            "analyzers_run": [],
        },
        "features": features,
        "tags": tags,
        "bn": {
            "nodes": nodes,
            "irr": irr,
        },
        "validations": validations,
    }