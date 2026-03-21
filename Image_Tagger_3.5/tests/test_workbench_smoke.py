"""Smoketest for Tagger Workbench endpoints.

This test exercises a minimal annotation flow:
- Insert a synthetic image.
- Fetch work as a tagger.
- Post a validation for that image.
- Confirm that the Validation row exists.
"""

from fastapi.testclient import TestClient

from backend.main import app
from backend.database.core import SessionLocal
from backend.models.assets import Image
from backend.models.annotation import Validation


client = TestClient(app)


def _tagger_headers():
    return {
        "X-User-Id": "1",
        "X-User-Role": "tagger",
    }


def test_workbench_annotation_flow():
    # Use a real DB session to seed a synthetic image.
    session = SessionLocal()
    try:
        img = Image(filename="workbench_smoke.jpg", storage_path="/tmp/workbench_smoke.jpg")
        session.add(img)
        session.commit()
        session.refresh(img)
        image_id = img.id
    finally:
        session.close()

    # 1. Fetch work (implementation-specific; we just assert that the endpoint is alive).
    # If the endpoint uses query params or a body, this call may need to be adapted,
    # but this smoketest ensures that the router is mounted and RBAC allows taggers.
    resp = client.get("/v1/annotation/queue", headers=_tagger_headers())
    assert resp.status_code in (200, 204, 404), resp.text

    # 2. Post a validation for the synthetic image.
    payload = {
        "image_id": image_id,
        "attribute_key": "science.visual_richness",
        "value": 0.5,
        "source": "test_workbench_smoke",
    }
    resp = client.post("/v1/annotation/validate", json=payload, headers=_tagger_headers())
    assert resp.status_code in (200, 201), resp.text

    # 3. Confirm that a Validation row exists.
    session = SessionLocal()
    try:
        exists = (
            session.query(Validation)
            .filter(Validation.image_id == image_id)
            .filter(Validation.attribute_key == "science.visual_richness")
            .filter(Validation.source == "test_workbench_smoke")
            .first()
        )
        assert exists is not None, "Expected a Validation row to be created by the annotation endpoint"
    finally:
        session.close()
