import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    message = Column(Text)
    is_bot = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = 'feedback'

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer)  # ID of the bot's message
    user_id = Column(String, index=True)
    is_positive = Column(Boolean)  # True for like, False for dislike
    correction = Column(Text, nullable=True)  # User's correction text
    document_ids = Column(String)  # Comma separated document IDs used for generation
    timestamp = Column(DateTime, default=datetime.utcnow)


DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///adaptive_rag.db')

# Module-level engine, built lazily (never at import time). Constructing the
# engine on import used to open a connection to the configured database, which
# made the module impossible to import in a test process without side effects.
engine = None


def make_engine(url: str | None = None):
    """Create a fresh SQLAlchemy engine for ``url`` (defaults to DATABASE_URL).

    Pure factory: it does not touch module state, so tests can spin up
    throwaway engines (e.g. ``sqlite:///:memory:``) in isolation.
    """
    url = url or DATABASE_URL
    # check_same_thread is a SQLite-only flag; other backends raise TypeError.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_engine(url: str | None = None):
    """Build the module-level engine and bind :data:`SessionLocal` to it."""
    global engine
    engine = make_engine(url)
    SessionLocal.configure(bind=engine)
    return engine


class _LazySessionLocal(sessionmaker):
    """``sessionmaker`` that binds its engine on first use.

    Keeps ``SessionLocal`` a drop-in session factory for existing callers while
    deferring engine construction until a session is actually requested.
    """

    def __call__(self, *args, **kwargs) -> Session:
        if engine is None:
            init_engine()
        return super().__call__(*args, **kwargs)


# Public session factory (back-compat name). Bound lazily via init_engine().
SessionLocal = _LazySessionLocal(autocommit=False, autoflush=False)


def init_db(url: str | None = None):
    """Ensure the engine exists, then create all tables."""
    if engine is None:
        init_engine(url)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
