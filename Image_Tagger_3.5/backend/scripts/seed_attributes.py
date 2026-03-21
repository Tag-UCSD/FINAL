"""
Seed script for attribute taxonomy.

Reads contracts/attributes.yml (v2.6.3-compatible format) and inserts
Attribute rows into the v3 database if they are missing.

Intended usage (from repo root):

    python -m backend.scripts.seed_attributes

In Docker (install.sh):

    docker-compose exec -T api python backend/scripts/seed_attributes.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

import yaml
from sqlalchemy.orm import Session

from backend.database.core import SessionLocal, engine
from backend.models import Base, Attribute


REPO_ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTES_YML = REPO_ROOT / "contracts" / "attributes.yml"


def parse_attributes(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parse the slightly non-standard attributes.yml shipped in v2.6.3.

    The file has a header:

        schema:
          - id: string
        # (additional schema fields omitted)
        attributes:
          - id: geometry.curvilinearity
            name: Curvilinearity

    followed by a long list of '- id:' blocks. We do a lightweight
    streaming parse that:

    - ignores the 'schema' section
    - starts collecting once we see 'attributes:'
    - treats any line starting with '- id:' as a new record
    - collects subsequent 'key: value' lines into the same record
    """
    entries: List[Dict[str, Any]] = []
    in_attrs = False
    current: Dict[str, Any] | None = None

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not in_attrs:
            if stripped.startswith("attributes:"):
                in_attrs = True
            continue

        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- id:"):
            if current:
                entries.append(current)
            current = {}
            _, _, val = stripped.partition(":")
            current["id"] = val.strip()
            continue

        if current is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            current[key] = val

    if current:
        entries.append(current)

    return entries


def seed() -> None:
    if not ATTRIBUTES_YML.exists():
        print(f"[seed_attributes] attributes.yml not found at {ATTRIBUTES_YML}")
        return

    lines = ATTRIBUTES_YML.read_text(encoding="utf-8").splitlines()
    raw_entries = parse_attributes(lines)
    print(f"[seed_attributes] Parsed {len(raw_entries)} attribute entries from YAML.")

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    created = 0
    try:
        for raw in raw_entries:
            key = raw.get("id")
            if not key:
                continue
            existing = db.query(Attribute).filter_by(key=key).first()
            if existing:
                continue

            attr = Attribute(
                key=key,
                name=raw.get("name", key),
                category=None,  # Could be inferred later from canonical tree
                level=raw.get("level"),
                range=raw.get("range"),
                sources=raw.get("sources"),
                notes=raw.get("notes"),
                is_active=True,
                source_version="v2.6.3",
            )
            db.add(attr)
            created += 1

        db.commit()
    finally:
        db.close()

    print(f"[seed_attributes] Created {created} new Attribute rows.")


if __name__ == "__main__":
    seed()