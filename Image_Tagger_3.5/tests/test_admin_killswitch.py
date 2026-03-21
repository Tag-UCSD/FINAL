import json

from fastapi.testclient import TestClient

from backend.main import app
from backend.database import SessionLocal, engine
from backend import models
from backend.models.config import ToolConfig

client = TestClient(app)


def _ensure_db():
    # Create tables if they do not exist.
    models.Base.metadata.create_all(bind=engine)


def test_admin_kill_switch_disables_paid_models():
    """Ensure the kill-switch endpoint toggles the global paid-model state.

    This test seeds at least one paid ToolConfig, verifies that the
    initial budget reflects `is_kill_switched == False`, then calls the
    kill-switch with `active=true` and asserts that no paid models remain
    enabled and that the returned BudgetStatus reports the kill switch
    as active.
    """
    _ensure_db()

    # Seed a paid tool config if none exist yet.
    with SessionLocal() as db:
        existing_paid = (
            db.query(ToolConfig)
            .filter(ToolConfig.cost_per_1k_tokens > 0.0)
            .count()
        )
        if existing_paid == 0:
            cfg = ToolConfig(
                name="unit_test_paid_model",
                provider="test",
                cost_per_1k_tokens=1.0,
                cost_per_image=0.0,
                is_enabled=True,
                settings={},
            )
            db.add(cfg)
            db.commit()

    headers = {"X-User-Role": "admin"}

    # Baseline: budget should be reachable and expose a boolean flag.
    resp_budget = client.get("/api/v1/admin/budget", headers=headers)
    assert resp_budget.status_code == 200
    data_budget = resp_budget.json()
    assert "is_kill_switched" in data_budget
    assert isinstance(data_budget["is_kill_switched"], bool)
    # With at least one paid model enabled, the kill switch should be off.
    assert data_budget["is_kill_switched"] is False

    # Activate the kill switch.
    resp_kill = client.post(
        "/api/v1/admin/kill-switch",
        headers=headers,
        params={"active": True},
    )
    assert resp_kill.status_code == 200
    data_kill = resp_kill.json()
    assert "is_kill_switched" in data_kill
    assert data_kill["is_kill_switched"] is True

    # All paid models should now be disabled in the DB.
    with SessionLocal() as db:
        remaining_paid = (
            db.query(ToolConfig)
            .filter(
                ToolConfig.cost_per_1k_tokens > 0.0,
                ToolConfig.is_enabled.is_(True),
            )
            .count()
        )
        assert remaining_paid == 0
