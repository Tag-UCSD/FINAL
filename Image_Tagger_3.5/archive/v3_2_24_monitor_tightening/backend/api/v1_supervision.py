from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from typing import List

from backend.database.core import get_db
from backend.services.auth import require_admin
from backend.models.users import User
from backend.models.annotation import Validation
from backend.models.assets import Image
from backend.schemas.supervision import TaggerPerformance, IRRStat, ValidationDetail


router = APIRouter(prefix="/v1/monitor", tags=["Supervisor Dashboard"])


@router.get("/velocity", response_model=List[TaggerPerformance])
async def get_tagger_velocity(
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> List[TaggerPerformance]:
    """Aggregate basic per-tagger velocity and status from Validation records.

    This endpoint powers the Team Velocity table in the Supervisor dashboard.
    """
    stmt = (
        select(
            Validation.user_id,
            func.count(Validation.id).label("count"),
            func.avg(Validation.duration_ms).label("avg_duration"),
        )
        .group_by(Validation.user_id)
    )
    rows = db.execute(stmt).all()
    if not rows:
        return []

    user_ids = [r.user_id for r in rows if r.user_id is not None]
    users_by_id: dict[int, User] = {}
    if user_ids:
        res = db.execute(select(User).where(User.id.in_(user_ids)))
        users_by_id = {u.id: u for u in res.scalars().all()}

    stats: List[TaggerPerformance] = []
    for r in rows:
        uid = r.user_id
        u = users_by_id.get(uid)
        username = u.username if u is not None else f"user-{uid}"

        avg_ms = int(r.avg_duration or 0)

        # Simple heuristic: if someone is making many decisions with
        # very low dwell time, flag them as suspicious.
        status = "active"
        if r.count and avg_ms > 0 and r.count >= 50 and avg_ms < 400:
            status = "flagged"

        stats.append(
            TaggerPerformance(
                user_id=uid,
                username=username,
                images_validated=int(r.count or 0),
                avg_duration_ms=avg_ms,
                status=status,
            )
        )

    return stats


@router.get("/irr", response_model=List[IRRStat])
async def get_irr_stats(
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> List[IRRStat]:
    """Compute a simple per-image agreement score across raters.

    This is *not* full Fleiss’ kappa. Instead it provides:
      - agreement_score: proportion of ratings in the majority (0–1)
      - conflict_count: min(#positives, #negatives) as a crude conflict indicator
      - raters: list of user identifiers contributing to the image

    This is enough to drive a meaningful IRR heatmap in the Supervisor GUI and
    can later be upgraded to true kappa without breaking the contract.
    """
    stmt = select(
        Validation.image_id,
        Validation.user_id,
        Validation.value,
    )
    rows = db.execute(stmt).all()
    if not rows:
        return []

    per_image: dict[int, list[tuple[int, float]]] = {}
    for image_id, user_id, value in rows:
        if image_id is None or user_id is None:
            continue
        per_image.setdefault(image_id, []).append((user_id, value))

    if not per_image:
        return []

    image_ids = list(per_image.keys())
    images_by_id: dict[int, Image] = {}
    if image_ids:
        res = db.execute(select(Image).where(Image.id.in_(image_ids)))
        images_by_id = {img.id: img for img in res.scalars().all()}

    irr_stats: List[IRRStat] = []
    # Limit to a reasonable sample for dashboard purposes
    for image_id, tuples in list(per_image.items())[:100]:
        values = [v for (_, v) in tuples if v is not None]
        if not values:
            continue

        # Binary discretisation: value > 0.5 → positive, else negative
        labels = [1 if v > 0.5 else 0 for v in values]
        n = len(labels)
        pos = sum(labels)
        neg = n - pos

        majority = max(pos, neg)
        agreement = majority / n if n > 0 else 0.0
        conflict_count = int(min(pos, neg))

        img = images_by_id.get(image_id)
        filename = img.filename if img is not None else f"image-{image_id}"

        raters = [str(uid) for (uid, _) in tuples]

        irr_stats.append(
            IRRStat(
                image_id=image_id,
                filename=filename,
                agreement_score=float(agreement),
                conflict_count=conflict_count,
                raters=raters,
            )
        )

    return irr_stats

@router.get("/image/{image_id}/validations", response_model=list[ValidationDetail])
async def get_image_validations(
    image_id: int,
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> list[ValidationDetail]:
    """Return all validations for a given image, joined with user info.

    This powers the Tag Inspector view in the Supervisor dashboard.
    """
    stmt = (
        select(
            Validation.id,
            Validation.user_id,
            Validation.attribute_key,
            Validation.value,
            Validation.duration_ms,
            Validation.created_at,
            User.username,
        )
        .join(User, User.id == Validation.user_id)
        .where(Validation.image_id == image_id)
        .order_by(Validation.created_at.desc())
    )
    rows = db.execute(stmt).all()
    results: list[ValidationDetail] = []
    for vid, uid, attr_key, value, duration_ms, created_at, username in rows:
        results.append(
            ValidationDetail(
                id=vid,
                user_id=uid,
                username=username or f"user-{uid}",
                attribute_key=attr_key,
                value=float(value) if value is not None else 0.0,
                duration_ms=duration_ms,
                created_at=created_at,
            )
        )
    return results