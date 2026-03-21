"""Audit script for VLM-derived attribute variance.

This script scans Validation records for cognitive.* and affect.*
attributes produced by the science pipeline / VLM, and flags cases
where the output distribution is suspiciously collapsed (e.g. the
same value for >90% of images).

Usage (inside the Docker `api` container)
-----------------------------------------

    python -m backend.scripts.audit_vlm_variance

The output is a JSON blob with a summary for each attribute key as
well as a list of suspicious keys that may indicate a failed prompt
or misconfigured VLM model.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session

from backend.database.core import SessionLocal
from backend.models.annotation import Validation


TARGET_PREFIXES: Tuple[str, ...] = ("cognitive.", "affect.")
MIN_COUNT_DEFAULT: int = 50
MODE_COLLAPSE_THRESHOLD_DEFAULT: float = 0.9


def _load_values_for_key(session: Session, key: str) -> List[float]:
    """Load numeric values for a given attribute key.

    We restrict to science_pipeline sources, which is where the VLM-backed
    cognitive / affective attributes land in the current architecture.
    """
    rows = (
        session.query(Validation.value)
        .filter(Validation.attribute_key == key)
        .filter(Validation.value.is_not(None))
        .filter(Validation.source.like("science_pipeline%"))
    )
    values: List[float] = []
    for (val,) in rows:
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue
    return values


def audit_vlm_variance(
    session: Session,
    min_count: int = MIN_COUNT_DEFAULT,
    mode_collapse_threshold: float = MODE_COLLAPSE_THRESHOLD_DEFAULT,
) -> Dict[str, Dict[str, float]]:
    """Compute variance diagnostics for VLM-backed attributes.

    Returns a mapping from attribute_key to a small summary dict containing:
    - count
    - mean
    - std
    - mode_value
    - mode_ratio
    - is_suspicious
    """
    report: Dict[str, Dict[str, float]] = {}

    # Discover candidate keys
    keys: List[str] = []
    for prefix in TARGET_PREFIXES:
        q = (
            session.query(Validation.attribute_key)
            .filter(Validation.attribute_key.like(f"{prefix}%"))
            .distinct()
        )
        keys.extend(k for (k,) in q)

    for key in sorted(set(keys)):
        values = _load_values_for_key(session, key)
        n = len(values)
        if n < min_count:
            continue

        rounded = [round(v, 2) for v in values]
        counts = Counter(rounded)
        if not counts:
            continue
        mode_value, mode_freq = counts.most_common(1)[0]
        mode_ratio = mode_freq / float(n)

        mean = sum(rounded) / float(n)
        var = sum((x - mean) ** 2 for x in rounded) / float(n)
        std = math.sqrt(var)

        is_suspicious = mode_ratio >= mode_collapse_threshold or std < 0.02

        report[key] = {
            "count": n,
            "mean": mean,
            "std": std,
            "mode_value": mode_value,
            "mode_ratio": mode_ratio,
            "is_suspicious": is_suspicious,
        }

    return report


def main() -> None:
    session = SessionLocal()
    try:
        report = audit_vlm_variance(session)
    finally:
        session.close()

    suspicious = {k: v for k, v in report.items() if v.get("is_suspicious")}
    payload = {
        "ok": not bool(suspicious),
        "suspicious_keys": suspicious,
        "all_keys": report,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
