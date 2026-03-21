# Image Tagger v3.4.63 — BN / DB Tightening + Optional Health Guard

Scope of this sprint
--------------------
- Tie BN-facing Validation rows more tightly to the canonical Attribute
  registry used throughout the system.
- Provide an optional, database-aware health check that can be run
  manually or via the governance guardian to detect drift between:
    * `validations.attribute_key`
    * `attributes.key`
    * BN candidate keys from the science index catalog.

1. Validation.attribute_key → ForeignKey(attributes.key)
--------------------------------------------------------

The `Validation` model in `backend/models/annotation.py` now declares
`attribute_key` as:

    ForeignKey("attributes.key")

This means every validation row (including those emitted by the science
pipeline and BN export) must reference a key present in the `attributes`
table in fresh schemas.

The model also includes an explicit relationship to `Attribute`:

    attribute = relationship("Attribute", back_populates="validations")

2. Attribute ↔ Validation relationship
--------------------------------------

The `Attribute` model in `backend/models/attribute.py` now exposes:

    validations = relationship("Validation", back_populates="attribute")

This gives a clean ORM join path both ways for BN snapshots, training
exports, and any downstream analytics that need to reason over the
attribute registry and its associated validation records.

3. Schema sanity test for the FK
--------------------------------

Added `tests/test_validation_attribute_fk_schema.py` to assert that the
`Validation` table includes a foreign key whose target is `attributes.key`.

This test is database-agnostic (it inspects SQLAlchemy table metadata)
and will fail fast if future refactors accidentally drop or rename the FK.

4. BN / DB health checker module
--------------------------------

New module: `backend/scripts/bn_db_health.py`

- Provides a function:

      run_health_check(exit_on_failure: bool = True) -> Dict[str, Any]

  that:

    * collects active `attributes.key` values;
    * collects distinct `Validation.attribute_key` values;
    * obtains BN candidate keys via `get_candidate_bn_keys()`; and
    * reports:

          - orphan Validation.attribute_key values (not in attributes.key)
          - BN candidate keys that are missing from attributes.key

- CLI usage (from repo root):

      python -m backend.scripts.bn_db_health

  or equivalently:

      python backend/scripts/bn_db_health.py

The CLI exits with status 1 if any violations are detected.

5. Optional integration into the governance guardian
----------------------------------------------------

- `scripts/guardian.py` now exposes `_check_bn_db_health`, which can be
  enabled via a new constraint:

      constraints:
        ...
        check_bn_db_health: false

- When `check_bn_db_health` is set to `true` in `v3_governance.yml` and
  `guardian.py verify` is run in an environment with a live database,
  the BN / DB health check will run in-process and report any violations
  as governance failures.

Operational notes
-----------------

- Existing databases created before v3.4.63 will not automatically gain
  the new FK constraint on `validations.attribute_key`. For production
  deployments you should either:

    * run an ALTER TABLE to add the FK constraint manually, or
    * recreate the database schema from scratch using the updated models.

- The existing `backend/scripts/seed_attributes.py` remains the
  recommended way to populate the `attributes` table before running
  science or BN export.

- The BN / DB health check requires a running database and an environment
  where `SessionLocal` is correctly configured (e.g., inside Docker via
  `docker-compose exec api ...`). By default the guardian constraint
  `check_bn_db_health` is `false` to avoid breaking installs that do not
  yet have a live DB.
