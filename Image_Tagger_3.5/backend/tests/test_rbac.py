"""
RBAC enforcement tests for Admin endpoints.
These tests avoid external DB dependencies by overriding get_db to in-memory SQLite.
"""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.database.core import Base, get_db


# In-memory SQLite for tests
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_admin_endpoint_forbidden_without_admin_header():
    # No headers → require_admin should reject
    r = client.get("/v1/admin/models")
    assert r.status_code in (401, 403)


def test_admin_endpoint_allows_admin_header():
    r = client.get("/v1/admin/models", headers={"X-User-Role": "admin"})
    assert r.status_code == 200