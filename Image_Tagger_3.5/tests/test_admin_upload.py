import io
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.database import SessionLocal, engine
from backend import models

client = TestClient(app)


def _ensure_db():
    # Create tables if they do not exist
    models.Base.metadata.create_all(bind=engine)


def test_admin_bulk_upload_creates_images(tmp_path):
    _ensure_db()
    # Minimal PNG header bytes; contents do not need to be a real image for this endpoint
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"fakeimagebytes"
    files = [
        ("files", ("test.png", io.BytesIO(png_bytes), "image/png")),
    ]
    headers = {"X-User-Role": "admin"}
    resp = client.post("/api/v1/admin/upload", files=files, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    # At least one image should be created
    assert data.get("created_count", 0) >= 1
    image_ids = data.get("image_ids", [])
    assert isinstance(image_ids, list)
    # Confirm that at least one of the returned IDs exists in the database
    with SessionLocal() as db:
        found = db.query(models.Image).filter(models.Image.id.in_(image_ids)).count()
        assert found >= 1
