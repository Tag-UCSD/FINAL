import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def _admin_headers():
    """Minimal admin identity for RBAC-protected endpoints."""
    return {
        "X-User-Id": "1",
        "X-User-Role": "admin",
    }


def _tagger_headers():
    """Minimal tagger identity for workbench / explorer endpoints."""
    return {
        "X-User-Id": "10",
        "X-User-Role": "tagger",
    }


def test_health_check():
    """Verify the system is alive and versioned correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"
    assert data.get("version") == "3.0.0"


def test_root_descriptor():
    """Root endpoint should advertise docs and workbench API entry point."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "workbench_api" in data


def test_admin_models_smoke():
    """Admin models endpoint should respond with a JSON list.

    This will fail if the Admin router is not mounted or if RBAC wiring is
    broken, which is the intended signal for CI.
    """
    response = client.get("/v1/admin/models", headers=_admin_headers())
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)


def test_admin_training_export_empty_list_ok():
    """Admin training export should accept an empty image_ids list.

    This exercise the TrainingExporter path without requiring any DB rows.
    """
    payload = {"image_ids": [], "format": "json"}
    response = client.post(
        "/v1/admin/training/export",
        headers=_admin_headers(),
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_explorer_export_empty_list_ok():
    """Explorer export should be wired and RBAC-protected for taggers.

    Like the admin export, this uses an empty image_ids list to avoid
    depending on seeded data while still verifying router + schema wiring.
    """
    payload = {"image_ids": [], "format": "json"}
    response = client.post(
        "/v1/explorer/export",
        headers=_tagger_headers(),
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_404_handling_json():
    """Invalid routes should return JSON 404, not HTML."""
    response = client.get("/v1/non_existent")
    assert response.status_code == 404
    body = response.json()
    assert isinstance(body, dict)
    assert body.get("detail") == "Not Found"


if __name__ == "__main__":
    print("Running basic API smoketests...")
    test_health_check()
    test_root_descriptor()
    print("Selected smoketests: PASS (if no assertion errors shown)")