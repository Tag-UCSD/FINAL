from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from typing import Generator
import os

# Configuration
# Defaulting to localhost postgres. In Docker, this will be overridden by ENV vars.
# NOTE: v3 Architecture explicitly drops support for SQLite to prevent locking in multi-user mode.
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://user:password@localhost:5432/image_tagger_v3"
)

# Engine Setup
# pool_pre_ping=True handles DB connection drops gracefully
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    """
    Unified SQLAlchemy Base for v3 Enterprise Refactor.
    All models must inherit from this class.
    """
    pass

def get_db() -> Generator:
    """
    FastAPI Dependency for database sessions.
    Ensures connections are closed after request lifecycle.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()