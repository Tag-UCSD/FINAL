from fastapi.testclient import TestClient

from backend.main import app
from backend.database import SessionLocal, engine
from backend import models

client = TestClient(app)


def _ensure_db():
    models.Base.metadata.create_all(bind=engine)


def _seed_image_and_validation():
    db = SessionLocal()
    try:
        image = models.Image(source="unit_test", path="test/path.png")
        db.add(image)
        db.commit()
        db.refresh(image)

        validation = models.Validation(
            image_id=image.id,
            tagger_id="tagger-1",
            tool_config_id="default",
            raw_payload={"foo": "bar"},
        )
        db.add(validation)
        db.commit()
        db.refresh(validation)
        return image.id
    finally:
        db.close()


def test_monitor_tag_inspector_returns_validations():
    _ensure_db()
    image_id = _seed_image_and_validation()

    headers = {"X-User-Role": "admin"}
    resp = client.get(f"/api/v1/monitor/image/{image_id}/validations", headers=headers)
    assert resp.status_code == 200
    payload = resp.json()
    # We expect at least one validation in the response
    assert isinstance(payload, list)
    assert len(payload) >= 1


def test_monitor_image_inspector_shape_and_rbac():
    """Smoketest for the Tag Inspector endpoint shape and RBAC.

    We do not require real science features or BN metadata here. The goal is to
    ensure the endpoint is wired, RBAC is enforced, and the payload has the expected
    top-level structure so that the frontend can rely on it.
    """
    _ensure_db()
    image_id = _seed_image_and_validation()

    # Without admin header, access should be denied.
    resp_forbidden = client.get(f"/api/v1/monitor/image/{image_id}/inspector")
    assert resp_forbidden.status_code in (401, 403)

    headers = {"X-User-Role": "admin"}
    resp_ok = client.get(f"/api/v1/monitor/image/{image_id}/inspector", headers=headers)
    assert resp_ok.status_code == 200

    payload = resp_ok.json()
    assert isinstance(payload, dict)

    # Basic shape checks.
    for key in ("image", "pipeline", "features", "tags", "bn", "validations"):
        assert key in payload, f"Missing '{key}' in inspector payload"

    assert isinstance(payload["image"], dict)
    assert isinstance(payload["pipeline"], dict)
    assert isinstance(payload["features"], list)
    assert isinstance(payload["tags"], list)
    assert isinstance(payload["bn"], dict)
    assert isinstance(payload["validations"], list)

    # Pipeline scaffold shape.
    pipeline = payload["pipeline"]
    assert "overall_status" in pipeline
    assert "analyzers_run" in pipeline
    assert isinstance(pipeline["analyzers_run"], list)

    # BN scaffold shape.
    bn = payload["bn"]
    assert "nodes" in bn
    assert "irr" in bn
    assert isinstance(bn["nodes"], list)
