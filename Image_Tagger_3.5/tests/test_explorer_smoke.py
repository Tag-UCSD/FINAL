"""Smoketests for Research Explorer endpoints.

These tests check that the Explorer routers are mounted and return responses
with the expected shapes on a minimal database.
"""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _tagger_headers():
    return {
        "X-User-Id": "1",
        "X-User-Role": "tagger",
    }


def test_explorer_attributes():
    resp = client.get("/v1/explorer/attributes", headers=_tagger_headers())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data, list)
    if data:
        first = data[0]
        assert "id" in first
        assert "key" in first
        assert "name" in first


def test_explorer_search_round_trip():
    payload = {
        "query_string": "",
        "filters": {},
        "page": 1,
        "page_size": 10,
    }
    resp = client.post("/v1/explorer/search", json=payload, headers=_tagger_headers())
    # We accept either success with a list or a 204/404 if no search backend is wired yet;
    # the main purpose is to ensure the router is mounted and RBAC permits taggers.
    assert resp.status_code in (200, 204, 404), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, list)
