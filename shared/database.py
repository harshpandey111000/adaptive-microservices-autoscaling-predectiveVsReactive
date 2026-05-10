"""SQLAlchemy engine/session setup.

All services use this shared session factory for the metrics database.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from shared.config import settings


engine = create_engine(settings.metrics_db_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
