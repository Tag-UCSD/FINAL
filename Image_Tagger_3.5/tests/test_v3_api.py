import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def _admin_headers():
    """Minimal admin identity for RBAC-protected endpoints."""
    return {
        "X-User-Id": "1",
        "X-User-Role": "admin",
        "X-Auth-Token": "dev_secret_key_change_me",  # Required for privileged roles
    }


def _tagger_headers():
    """Minimal tagger identity for RBAC-protected endpoints."""
    return {
        "X-User-Id": "2",
        "X-User-Role": "tagger",
    }


def test_health_root():
    c = TestClient(app)
    resp = c.get("/")
    assert resp.status_code == 200


def test_health_endpoint():
    c = TestClient(app)
    resp = c.get("/health")
    assert resp.status_code in (200, 204)


def test_admin_models_rbac():
    c = TestClient(app)
    resp_forbidden = c.get("/v1/admin/models")
    assert resp_forbidden.status_code in (401, 403)

    resp_ok = c.get("/v1/admin/models", headers=_admin_headers())
    assert resp_ok.status_code in (200, 204)


def test_explorer_attributes():
    c = TestClient(app)
    resp = c.get("/v1/explorer/attributes", headers=_tagger_headers())
    assert resp.status_code in (200, 204)


def test_explorer_search_smoketest():
    c = TestClient(app)
    resp = c.post(
        "/v1/explorer/search",
        json={"filters": {}, "page": 1, "page_size": 5},
        headers=_tagger_headers(),
    )
    assert resp.status_code in (200, 204)


def test_explorer_export_empty_list():
    c = TestClient(app)
    resp = c.post(
        "/v1/explorer/export",
        json={"image_ids": []},
        headers=_tagger_headers(),
    )
    assert resp.status_code in (200, 204)


def test_monitor_velocity_rbac_and_shape():
    c = TestClient(app)

    # No headers → forbidden
    resp_forbidden = c.get("/v1/monitor/velocity")
    assert resp_forbidden.status_code in (401, 403)

    # Admin headers → OK, returns list (possibly empty)
    resp_ok = c.get("/v1/monitor/velocity", headers=_admin_headers())
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert isinstance(data, list)


def test_monitor_irr_rbac_and_shape():
    c = TestClient(app)

    resp_forbidden = c.get("/v1/monitor/irr")
    assert resp_forbidden.status_code in (401, 403)

    resp_ok = c.get("/v1/monitor/irr", headers=_admin_headers())
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert isinstance(data, list)