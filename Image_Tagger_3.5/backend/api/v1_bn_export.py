"""BN export API for Image Tagger v3.

This router exposes a minimal endpoint to export BN-ready rows that contain
science indices and their bins for each image, based on the Validation table.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.core import get_db
from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.schemas.bn_export import BNRow, BNCodebook, BNVariable, BNValidationRow
from backend.science.index_catalog import get_candidate_bn_keys, get_index_metadata

router = APIRouter(tags=["bn_export"], prefix="/v1/export")


BIN_LABELS = {0.0: "low", 1.0: "mid", 2.0: "high"}
# Also support integer keys just in case
BIN_LABELS.update({0: "low", 1: "mid", 2: "high"})


def _collect_indices_for_image(
    session: Session,
    image_id: int,
    keys: List[str],
) -> Dict[str, Optional[float]]:
    """Collect continuous index values for the given image from Validation.

    We treat the latest Validation per (image_id, attribute_key, source) as the
    canonical value, restricted to entries produced by the science pipeline.
    """
    indices: Dict[str, Optional[float]] = {k: None for k in keys}

    q = (
        session.query(Validation)
        .filter(Validation.image_id == image_id)
        .filter(Validation.attribute_key.in_(keys))
        .filter(Validation.source.like("science_pipeline%"))
    )

    for row in q:
        key = row.attribute_key
        if key in indices and row.value is not None:
            # Last write wins; this keeps the implementation simple.
            indices[key] = float(row.value)

    return indices


def _collect_bins_for_image(session: Session, image_id: int) -> Dict[str, Optional[str]]:
    """Collect bin labels for composite indices from Validation.

    The underlying science pipeline currently stores bins as numeric codes
    (0, 1, 2). We map these to string labels (low, mid, high) for BN and UX.
    """
    bins: Dict[str, Optional[str]] = {}

    metadata = get_index_metadata()
    bin_keys: List[str] = []
    for _, info in metadata.items():
        binspec = info.get("bins")
        if not binspec:
            continue
        field = binspec.get("field")
        if field:
            bin_keys.append(field)

    if not bin_keys:
        return bins

    q = (
        session.query(Validation)
        .filter(Validation.image_id == image_id)
        .filter(Validation.attribute_key.in_(bin_keys))
        .filter(Validation.source.like("science_pipeline%"))
    )

    for row in q:
        key = row.attribute_key
        raw = row.value
        label: Optional[str] = None
        if raw is not None:
            label = BIN_LABELS.get(raw)
        bins[key] = label

    # Ensure all expected bin keys are present in the dict
    for k in bin_keys:
        bins.setdefault(k, None)

    return bins




def _compute_irr_for_image(session: Session, image_id: int) -> Optional[float]:
    """Compute a simple IRR score for a given image across all attributes.

    This mirrors the pairwise-agreement logic used in the Supervisor's /irr
    endpoint, but without any time-window filter. For each attribute with
    at least two validations, we compute the fraction of agreeing pairs of
    ratings, then average across attributes.

    Returns None if there is insufficient overlapping data.
    """
    # Fetch all validations for this image (human + pipeline). We do not
    # filter by source here; downstream BN tools can decide how to treat
    # low-IRR images.
    from collections import defaultdict

    rows = session.query(
        Validation.attribute_key,
        Validation.value,
    ).filter(Validation.image_id == image_id).all()

    if not rows:
        return None

    grouped: Dict[str, list] = defaultdict(list)
    for attr_key, value in rows:
        grouped[attr_key].append(value)

    scores = []
    for _attr_key, values in grouped.items():
        n = len(values)
        if n < 2:
            continue
        total_pairs = 0
        agree_pairs = 0
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                total_pairs += 1
                if vi == vj:
                    agree_pairs += 1

        if total_pairs == 0:
            continue

        scores.append(agree_pairs / total_pairs)

    if not scores:
        return None

    return float(sum(scores) / len(scores))


def _bin_irr(score: Optional[float]) -> Optional[str]:
    """Map an IRR score into a coarse bin.

    This is intentionally coarse so BN tools can quickly filter to
    "high-consensus" images (e.g., irr_bin == "high").
    """
    if score is None:
        return None
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


@router.get("/bn-snapshot", response_model=List[BNRow])
def export_bn_snapshot(db: Session = Depends(get_db)) -> List[BNRow]:
    """Export a BN-ready snapshot of science indices and bins for each image.

    This endpoint reads from the Validation table, restricted to entries whose
    `source` starts with "science_pipeline". It is intended as a stable,
    inspectable contract for downstream BN tools, not as a full-featured data
    warehouse API.
    """
    candidate_keys = get_candidate_bn_keys()

    # Collect all image IDs
    image_ids: List[int] = [row.id for row in db.query(Image.id)]

    rows: List[BNRow] = []
    for image_id in image_ids:
        indices = _collect_indices_for_image(db, image_id, candidate_keys)
        bins = _collect_bins_for_image(db, image_id)
        irr = _compute_irr_for_image(db, image_id)
        irr_bin = _bin_irr(irr)

        rows.append(
            BNRow(
                image_id=image_id,
                source="image_tagger_v3.4.65",
                indices=indices,
                bins=bins,
                agreement_score=irr,
                irr_bin=irr_bin,
            )
        )

    return rows




@router.get("/bn-validations", response_model=List[BNValidationRow])
def export_bn_validations(db: Session = Depends(get_db)) -> List[BNValidationRow]:
    """Export one row per Validation record for hierarchical models.

    This endpoint exposes a flattened view over the Validation table
    for science-pipeline and manual sources, allowing downstream
    tools to model individual tagger bias, learning curves, and
    multi-level structures.
    """
    rows: List[BNValidationRow] = []

    q = (
        db.query(
            Validation.image_id,
            Validation.user_id,
            Validation.attribute_key,
            Validation.value,
            Validation.source,
            Validation.duration_ms,
        )
        .filter(
            Validation.source.like("science_pipeline%")
            | (Validation.source == "manual")
        )
    )

    for image_id, user_id, key, value, source, duration_ms in q:
        if value is None:
            continue
        rows.append(
            BNValidationRow(
                image_id=image_id,
                user_id=user_id,
                attribute_key=key,
                value=float(value),
                source=source,
                duration_ms=duration_ms or 0,
            )
        )

    return rows


@router.get("/bn-codebook", response_model=BNCodebook)
def get_bn_codebook() -> BNCodebook:
    """Return a codebook describing the BN export variables.

    This endpoint exposes the domain of each variable emitted by the
    `/v1/export/bn-snapshot` endpoint so that downstream BN /
    probabilistic programming tools do not need to guess at data types
    or valid states.
    """
    index_metadata = get_index_metadata()
    candidate_keys = get_candidate_bn_keys()

    variables: List[BNVariable] = []

    # Continuous index variables (science indices used as BN inputs)
    for key in candidate_keys:
        info = index_metadata.get(key, {})
        label = info.get("label", key)
        description = info.get("description", "")
        variables.append(
            BNVariable(
                name=key,
                label=label,
                description=description,
                role="index",
                var_type="continuous",
                states=None,
            )
        )

    # Bin variables derived from index metadata
    bin_fields: Dict[str, List[str]] = {}
    for _key, info in index_metadata.items():
        binspec = info.get("bins")
        if not binspec:
            continue
        field = binspec.get("field")
        if not field:
            continue
        values = binspec.get("values") or ["low", "mid", "high"]
        bin_fields[field] = list(values)

    for field, values in sorted(bin_fields.items()):
        variables.append(
            BNVariable(
                name=field,
                label=field,
                description=f"Bin for {field}",
                role="bin",
                var_type="ordinal",
                states=values,
            )
        )

    # Inter-rater reliability variables
    variables.append(
        BNVariable(
            name="agreement_score",
            label="agreement_score",
            description=(
                "Average inter-rater agreement for this image (0–1), "
                "computed as mean pairwise agreement across attributes."
            ),
            role="meta",
            var_type="continuous",
            states=None,
        )
    )
    variables.append(
        BNVariable(
            name="irr_bin",
            label="irr_bin",
            description=(
                "Coarse IRR bin derived from agreement_score: "
                "low / medium / high."
            ),
            role="meta",
            var_type="ordinal",
            states=["low", "medium", "high"],
        )
    )

    # Identifier + provenance variables
    variables.append(
        BNVariable(
            name="image_id",
            label="image_id",
            description="Numeric identifier of the image.",
            role="id",
            var_type="discrete",
            states=None,
        )
    )
    variables.append(
        BNVariable(
            name="source",
            label="source",
            description=(
                "Origin of this BN row (typically an Image Tagger "
                "version string)."
            ),
            role="meta",
            var_type="discrete",
            states=None,
        )
    )

    return BNCodebook(variables=variables)

