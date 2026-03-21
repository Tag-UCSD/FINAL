"""
Legacy prefix health checks for Image Tagger backend.
"""

from fastapi.testclient import TestClient

from backend.main import app
from backend.versioning import VERSION


client = TestClient(app)


def test_health_endpoint_with_legacy_prefix():
    response = client.get("/api/v1/tagger/health")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == VERSION


def test_health_endpoint_with_api_prefix():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == VERSION
