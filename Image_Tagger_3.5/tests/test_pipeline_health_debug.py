from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_pipeline_health_basic_shape():
    """Smoketest for /api/v1/debug/pipeline_health.

    We only assert the high-level response shape so the test is robust to
    future changes in the exact analyzer set or warning messages.
    """
    resp = client.get("/api/v1/debug/pipeline_health")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, dict)

    # Required top-level fields
    assert "import_ok" in data
    assert isinstance(data["import_ok"], bool)

    assert "cv2_available" in data
    assert isinstance(data["cv2_available"], bool)

    assert "analyzers_by_tier" in data
    assert isinstance(data["analyzers_by_tier"], dict)

    # Optional lists
    for key in ("warnings", "analyzer_errors"):
        if key in data:
            assert isinstance(data[key], list)

    # Basic analyzer entry shape (if any analyzers are present)
    analyzers_by_tier = data["analyzers_by_tier"]
    assert isinstance(analyzers_by_tier, dict)
    for tier, analyzers in analyzers_by_tier.items():
        assert isinstance(analyzers, list)
        for a in analyzers:
            assert isinstance(a, dict)
            assert "name" in a
            assert "tier" in a
            assert "requires" in a
            assert "provides" in a
