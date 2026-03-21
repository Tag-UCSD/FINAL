"""Smoketest for the BN export endpoint.

This test does *not* require the full science pipeline to run. Instead, it inserts a
synthetic image and a couple of Validation rows that mimic science output, then calls
the BN export function directly and checks that we see non-empty indices and bins.
"""

from backend.api.v1_bn_export import export_bn_snapshot
from backend.database.core import SessionLocal
from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.index_catalog import get_candidate_bn_keys, get_index_metadata


def test_bn_export_smoke():
    # This is an integration-style smoke test: it uses the real SessionLocal.
    session = SessionLocal()
    try:
        # 1. Create a synthetic image.
        img = Image(filename="bn_smoke_test.jpg", storage_path="/tmp/bn_smoke_test.jpg")
        session.add(img)
        session.commit()
        session.refresh(img)

        image_id = img.id

        # 2. Pick at least one candidate continuous index key.
        candidate_keys = get_candidate_bn_keys()
        assert candidate_keys, "Index catalog returned no candidate BN keys"
        idx_key = candidate_keys[0]

        # 3. Pick one bin field from the index metadata, if available.
        metadata = get_index_metadata()
        bin_key = None
        for _, info in metadata.items():
            binspec = info.get("bins")
            if binspec and "field" in binspec:
                bin_key = binspec["field"]
                break

        # 4. Insert synthetic Validation rows for the chosen keys.
        session.add(
            Validation(
                image_id=image_id,
                attribute_key=idx_key,
                value=0.7,
                source="science_pipeline_test",
            )
        )
        if bin_key is not None:
            session.add(
                Validation(
                    image_id=image_id,
                    attribute_key=bin_key,
                    value=2.0,
                    source="science_pipeline_test",
                )
            )
        session.commit()

        # 5. Call the BN export routine directly.
        rows = export_bn_snapshot(db=session)
        assert rows, "BN export returned no rows"

        # Find our image row.
        row = next((r for r in rows if r.image_id == image_id), None)
        assert row is not None, "BN export did not include the synthetic image"

        # 6. Continuous index should be non-None.
        assert idx_key in row.indices, "Candidate index key missing from BNRow.indices"
        assert row.indices[idx_key] is not None

        # 7. If we had a bin key, its label should be 'high' (2 -> 'high').
        if bin_key is not None:
            assert bin_key in row.bins, "Bin key missing from BNRow.bins"
            assert row.bins[bin_key] == "high"
    finally:
        session.close()
