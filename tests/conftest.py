"""Shared test fixtures.

These tests intentionally exercise only the modules that depend on SQLAlchemy
(``database`` and ``feedback``). Heavy optional dependencies (aiogram,
langchain, chromadb) are never imported here, so the suite runs with just
``pytest`` and ``sqlalchemy`` installed.
"""
import pytest

from database import Base, init_engine


@pytest.fixture(autouse=True)
def in_memory_db():
    """Bind an isolated in-memory SQLite database for every test."""
    engine = init_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
