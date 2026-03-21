"""Schema-level sanity check for Validation.attribute_key foreign key.

This test ensures that the SQLAlchemy model for Validation declares a
foreign key from ``validations.attribute_key`` to ``attributes.key``.
It does not require a running database.
"""

from backend.models.annotation import Validation


def test_validation_attribute_key_has_fk_to_attributes() -> None:
    fk_targets = {fk.target_fullname for fk in Validation.__table__.foreign_keys}
    assert "attributes.key" in fk_targets, fk_targets
