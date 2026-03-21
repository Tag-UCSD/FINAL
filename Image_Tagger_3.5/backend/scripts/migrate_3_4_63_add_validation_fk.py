"""Migration helper: add Validation.attribute_key → attributes.key foreign key.

This script is intended for PostgreSQL databases that were created
*before* v3.4.63, when the Validation.attribute_key column was not
yet backed by a real FOREIGN KEY in the live schema.

For databases initialised via v3.4.63+ with Base.metadata.create_all,
the constraint should already exist and this script will be a no-op.

Usage (inside the Docker `api` container)
----------------------------------------

    python -m backend.scripts.migrate_3_4_63_add_validation_fk

The script is idempotent and safe to run multiple times.
"""
from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from backend.database.core import engine
from backend.models.annotation import Validation


CONSTRAINT_NAME = "fk_validations_attribute_key_attributes_key"


def _has_fk() -> bool:
    """Return True if the Validation.attribute_key FK already exists.

    This uses SQLAlchemy's inspection API so it does not depend on
    any specific constraint name; it will accept either the canonical
    name defined here or an automatically-generated name, as long as
    the constrained and referred columns match.
    """
    inspector = inspect(engine)
    table_name = Validation.__table__.name  # type: ignore[attr-defined]
    fks: List[Dict[str, Any]] = inspector.get_foreign_keys(table_name)

    for fk in fks:
        constrained = fk.get("constrained_columns") or []
        referred_table = fk.get("referred_table")
        referred_cols = fk.get("referred_columns") or []
        name = fk.get("name") or ""

        if constrained == ["attribute_key"] and referred_table == "attributes" and referred_cols == ["key"]:
            return True
        if name == CONSTRAINT_NAME:
            return True

    return False


def _add_fk() -> None:
    """Issue the ALTER TABLE statement to add the FK.

    This assumes PostgreSQL semantics and will run inside a short-lived
    transaction. If the constraint already exists, PostgreSQL will
    raise an error and the caller will handle it.
    """
    table_name = Validation.__table__.name  # type: ignore[attr-defined]
    ddl = text(
        f"ALTER TABLE {table_name} "
        f"ADD CONSTRAINT {CONSTRAINT_NAME} "
        "FOREIGN KEY (attribute_key) REFERENCES attributes(key);"
    )

    with engine.begin() as conn:
        conn.execute(ddl)


def main(exit_on_failure: bool = True) -> int:
    """Entry point for the migration.

    Parameters
    ----------
    exit_on_failure:
        When True (default), a non-zero exit code will trigger
        ``sys.exit(code)`` in the CLI wrapper. When False, the caller
        (e.g. a higher-level script) can inspect the returned code.
    """
    try:
        if _has_fk():
            print(
                "[migrate_3_4_63_add_validation_fk] "
                "Foreign key already present; no migration needed."
            )
            return 0

        print(
            "[migrate_3_4_63_add_validation_fk] "
            "Adding foreign key on Validation.attribute_key → attributes.key..."
        )
        _add_fk()
        print("[migrate_3_4_63_add_validation_fk] Migration complete.")
        return 0
    except SQLAlchemyError as exc:
        msg = (
            "[migrate_3_4_63_add_validation_fk] Database error while applying migration: "
            f"{exc}"
        )
        print(msg)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected runtime errors
        msg = (
            "[migrate_3_4_63_add_validation_fk] Unexpected error: "
            f"{exc}"
        )
        print(msg)
        return 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Add the missing Validation.attribute_key → attributes.key foreign key "
            "for databases created before v3.4.63."
        )
    )
    parser.add_argument(
        "--no-exit-on-failure",
        action="store_true",
        help=(
            "Return a non-zero status code instead of calling sys.exit "
            "when an error occurs."
        ),
    )
    args = parser.parse_args()
    code = main(exit_on_failure=not args.no_exit_on_failure)
    if code != 0:
        sys.exit(code)
