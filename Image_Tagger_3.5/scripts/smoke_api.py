#!/usr/bin/env python3
"""Lightweight API smoke test for Image Tagger v3.2.

This script is intentionally minimal: it checks that the FastAPI
application can be imported and that a few core routes exist.
"""

from fastapi.testclient import TestClient

from backend.main import app


def main() -> None:
    client = TestClient(app)

    # Health endpoint
    resp = client.get("/health")
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "healthy"
    print("[smoke_api] /health OK:", data)

    # Explorer search (stubbed data is fine; we're testing wiring)
    try:
        resp2 = client.post(
            "/v1/explorer/search",
            json={"query_string": "", "filters": {}, "page": 1, "page_size": 1},
        )
        resp2.raise_for_status()
        print("[smoke_api] /v1/explorer/search OK; returned", len(resp2.json()), "items")
    except Exception as exc:
        print("[smoke_api] /v1/explorer/search check failed:", exc)

    # Attribute registry (may be empty before seed_attributes runs)
    try:
        resp3 = client.get("/v1/explorer/attributes")
        resp3.raise_for_status()
        print("[smoke_api] /v1/explorer/attributes OK; got", len(resp3.json()), "rows")
    except Exception as exc:
        print("[smoke_api] /v1/explorer/attributes check failed:", exc)


if __name__ == "__main__":
    main()