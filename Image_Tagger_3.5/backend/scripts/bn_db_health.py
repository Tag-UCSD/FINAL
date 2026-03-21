"""BN / DB health checker.

This module inspects the live database to ensure that:

- All Validation.attribute_key values are present in the canonical
  Attribute registry (attributes.key).
- All BN candidate keys exposed by the science index catalog are
  also present (and typically active) in the Attribute registry.

It is intended to be run either:

  - as a standalone CLI:

        python -m backend.scripts.bn_db_health

    (or equivalently: ``python backend/scripts/bn_db_health.py``)

  - or via the governance guardian when
    ``constraints.check_bn_db_health`` is enabled in v3_governance.yml.
"""
from __future__ import annotations

from typing import Any, Dict, Set

from sqlalchemy.exc import SQLAlchemyError

from backend.database.core import SessionLocal
from backend.models.attribute import Attribute
from backend.models.annotation import Validation
from backend.science.index_catalog import get_candidate_bn_keys


def _fetch_attribute_keys() -> Set[str]:
    db = SessionLocal()
    try:
        rows = db.query(Attribute.key).filter(Attribute.is_active.is_(True)).all()
        return {row[0] for row in rows if row[0] is not None}
    finally:
        db.close()


def _fetch_validation_keys() -> Set[str]:
    db = SessionLocal()
    try:
        rows = db.query(Validation.attribute_key).distinct().all()
        return {row[0] for row in rows if row[0] is not None}
    finally:
        db.close()


def run_health_check(exit_on_failure: bool = True) -> Dict[str, Any]:
    """Run BN / DB health checks.

    Parameters
    ----------
    exit_on_failure:
        When True (CLI usage), the process will exit with status 1 if any
        violations are detected. When False (guardian usage), a summary
        dictionary is returned and no ``sys.exit`` is called.

    Returns
    -------
    summary:
        Dict containing keys:

        - ok: bool
        - orphan_validations: int
        - missing_candidates: int
        - orphan_validation_keys: optional list[str] (truncated)
        - missing_candidate_keys: optional list[str] (truncated)
        - error: optional str if the check could not be completed
    """
    import sys

    summary: Dict[str, Any] = {
        "ok": False,
        "orphan_validations": 0,
        "missing_candidates": 0,
    }

    try:
        attr_keys = _fetch_attribute_keys()
        validation_keys = _fetch_validation_keys()
    except SQLAlchemyError as exc:  # pragma: no cover - depends on DB wiring
        msg = f"[bn_db_health] Database error while fetching keys: {exc}"
        print(msg)
        summary["error"] = msg
        if exit_on_failure:
            sys.exit(1)
        return summary

    # BN candidates are the keys we expect to show up in BN exports.
    candidate_keys = set(get_candidate_bn_keys())

    orphan_validation_keys = sorted(validation_keys - attr_keys)
    missing_candidate_keys = sorted(candidate_keys - attr_keys)

    summary["orphan_validations"] = len(orphan_validation_keys)
    summary["missing_candidates"] = len(missing_candidate_keys)
    summary["ok"] = not orphan_validation_keys and not missing_candidate_keys

    print("[bn_db_health] Attribute keys in registry:", len(attr_keys))
    print("[bn_db_health] Distinct Validation.attribute_key values:", len(validation_keys))
    print("[bn_db_health] BN candidate keys:", len(candidate_keys))

    if orphan_validation_keys:
        preview = ", ".join(orphan_validation_keys[:10])
        print(
            f"[bn_db_health] Orphan Validation.attribute_key values "
            f"(not in attributes.key): {len(orphan_validation_keys)} "
            f"(e.g., {preview})"
        )
        summary["orphan_validation_keys"] = orphan_validation_keys[:50]

    if missing_candidate_keys:
        preview = ", ".join(missing_candidate_keys[:10])
        print(
            f"[bn_db_health] BN candidate keys missing from attributes.key: "
            f"{len(missing_candidate_keys)} (e.g., {preview})"
        )
        summary["missing_candidate_keys"] = missing_candidate_keys[:50]

    if summary["ok"]:
        print("[bn_db_health] OK: No orphan validations and all BN candidate keys present.")
    else:
        print("[bn_db_health] FAIL: See details above.")

    if exit_on_failure and not summary["ok"]:
        sys.exit(1)

    return summary


def main() -> None:  # pragma: no cover - thin CLI wrapper
    run_health_check(exit_on_failure=True)


if __name__ == "__main__":
    main()
