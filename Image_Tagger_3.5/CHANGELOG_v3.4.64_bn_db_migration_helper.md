# Changelog v3.4.64 – BN/DB Migration Helper and Docs

- Added ``backend/scripts/migrate_3_4_63_add_validation_fk.py``, an idempotent
  helper for adding the missing ``Validation.attribute_key → attributes.key``
  foreign key on legacy PostgreSQL databases created before v3.4.63.
- Extended ``docs/devops_quickstart.md`` with a new section on BN / DB health
  checks and the legacy FK migration workflow, including example Docker
  commands for TAs and developers.
- No changes to the science pipeline, API surface, or frontend behaviour.
  Fresh installs of v3.4.63+ remain the canonical baseline; v3.4.64 is a
  documentation and operations refinement for teams with existing databases.
