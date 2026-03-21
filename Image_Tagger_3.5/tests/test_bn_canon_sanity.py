"""Canon sanity tests for BN export and index catalog.

These tests are higher-level than `test_bn_export_smoke` and focus on:

- Ensuring the index catalog entries for candidate BN inputs are well-formed.
- Ensuring the BN export uses exactly the candidate keys and expected bin fields.
"""

from backend.api.v1_bn_export import export_bn_snapshot
from backend.database.core import SessionLocal
from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.index_catalog import get_candidate_bn_keys, get_index_metadata


def test_index_catalog_candidate_entries_are_well_formed():
    """All candidate BN index entries should have labels, descriptions, types, and bins.

    This is a pure-catalog sanity check (no DB). It ensures that the BN-facing
    index definitions are complete and consistent enough to be used downstream.
    """
    metadata = get_index_metadata()
    candidate_keys = get_candidate_bn_keys()
    assert candidate_keys, "Index catalog returned no candidate BN keys"

    for key in candidate_keys:
        assert key in metadata, f"Candidate key {key!r} missing from index metadata"
        info = metadata[key]

        # Required fields
        assert info.get("label"), f"Index {key} missing label"
        assert info.get("description"), f"Index {key} missing description"
        assert info.get("type") in {"float", "int", "str"}, f"Index {key} has invalid type"

        # Bins are required for BN inputs in this design.
        binspec = info.get("bins")
        assert binspec is not None, f"Index {key} missing bins spec"
        assert "field" in binspec and binspec["field"], f"Index {key} bins spec missing field"
        assert "values" in binspec and isinstance(binspec["values"], list), f"Index {key} bins spec missing values"

        values = binspec["values"]
        assert values, f"Index {key} bins spec has empty values"
        # If we only have three levels, they should be ordered low/mid/high.
        if len(values) == 3:
            assert set(values) == {"low", "mid", "high"}, f"Index {key} bins values are unexpected: {values}"


def test_bn_export_respects_index_catalog_canon():
    """BN rows should expose indices and bins consistent with the index catalog.

    We seed synthetic Validation rows for *all* candidate indices and their
    bin fields, then call the export function and check that:

    - BNRow.indices contains exactly the candidate keys and all are non-None.
    - BNRow.bins contains the expected bin fields and labels.
    """
    metadata = get_index_metadata()
    candidate_keys = get_candidate_bn_keys()
    assert candidate_keys, "Index catalog returned no candidate BN keys"

    # Derive expected bin fields for candidate indices
    expected_bin_fields = set()
    for key in candidate_keys:
        binspec = metadata[key].get("bins")
        if binspec and "field" in binspec:
            expected_bin_fields.add(binspec["field"])

    session = SessionLocal()
    try:
        # Create a synthetic image
        img = Image(filename="bn_canon_test.jpg", storage_path="/tmp/bn_canon_test.jpg")
        session.add(img)
        session.commit()
        session.refresh(img)
        image_id = img.id

        # Seed continuous values for all candidate indices
        for idx_key in candidate_keys:
            session.add(
                Validation(
                    image_id=image_id,
                    attribute_key=idx_key,
                    value=0.5,
                    source="science_pipeline_test",
                )
            )

        # Seed bin codes (2 -> 'high') for all expected bin fields
        for bin_field in expected_bin_fields:
            session.add(
                Validation(
                    image_id=image_id,
                    attribute_key=bin_field,
                    value=2.0,
                    source="science_pipeline_test",
                )
            )

        session.commit()

        # Export BN snapshot
        rows = export_bn_snapshot(db=session)
        assert rows, "BN export returned no rows"

        # Find our image row
        row = next((r for r in rows if r.image_id == image_id), None)
        assert row is not None, "BN export did not include the synthetic image"

        # Indices canon: keys should match candidate index keys and be non-None
        assert set(row.indices.keys()) == set(candidate_keys)
        for key in candidate_keys:
            assert row.indices[key] is not None, f"Index {key} is unexpectedly None in BNRow.indices"

        # Bins canon: we at least expect all configured bin fields to appear
        for bin_field in expected_bin_fields:
            assert bin_field in row.bins, f"Bin field {bin_field!r} missing from BNRow.bins"
            # Where we used 2.0, we expect 'high' when low/mid/high is configured.
            # The label mapping is tested more directly in test_bn_export_smoke.
            if row.bins[bin_field] is not None:
                assert row.bins[bin_field] in {"low", "mid", "high"}
    finally:
        session.close()
