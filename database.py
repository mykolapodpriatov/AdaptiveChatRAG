import os
from datetime import datetime
from typing import cast

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    message = Column(Text)
    is_bot = Column(Boolean, default=False)
    # Comma-separated source document IDs returned by generate_response() for
    # bot messages, so a later 👎 on this message can thread the real IDs into
    # save_feedback() instead of an empty list. Unused for user messages.
    document_ids = Column(String)
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
engine: Engine | None = None


def make_engine(url: str | None = None) -> Engine:
    """Create a fresh SQLAlchemy engine for ``url`` (defaults to DATABASE_URL).

    Pure factory: it does not touch module state, so tests can spin up
    throwaway engines (e.g. ``sqlite:///:memory:``) in isolation.
    """
    url = url or DATABASE_URL
    # check_same_thread is a SQLite-only flag; other backends raise TypeError.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    # :memory: is per-connection unless we pin a single StaticPool connection,
    # which FastAPI TestClient needs (the endpoint runs on a worker thread).
    kwargs: dict = {}
    if url.startswith("sqlite") and ":memory:" in url:
        kwargs["poolclass"] = StaticPool
    return create_engine(url, connect_args=connect_args, **kwargs)


def init_engine(url: str | None = None) -> Engine:
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


def init_db(url: str | None = None) -> None:
    """Ensure the engine exists, then create all tables."""
    active = engine if engine is not None else init_engine(url)
    Base.metadata.create_all(bind=active)


def compute_feedback_stats(db: Session) -> dict[str, int]:
    """Return feedback counts bucketed by the ``is_positive`` flag.

    ``positive`` counts rows where ``is_positive`` is True and ``negative``
    where it is False; NULL values land in their own ``unknown`` bucket instead
    of being silently miscounted as negative. ``total_feedback`` always equals
    ``positive + negative + unknown``.
    """
    total = db.query(Feedback).count()
    positive = db.query(Feedback).filter(Feedback.is_positive.is_(True)).count()
    negative = db.query(Feedback).filter(Feedback.is_positive.is_(False)).count()
    unknown = total - positive - negative
    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "unknown": unknown,
    }


def fetch_session_history(db: Session, session_id: str) -> list[ChatHistory]:
    """Return a session's chat history ordered oldest-first."""
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.timestamp)
        .all()
    )


def encode_document_ids(document_ids: list) -> str:
    """Serialize document ids into the comma-separated form stored on both
    ``ChatHistory.document_ids`` and ``Feedback.document_ids``.
    """
    return ",".join(str(doc_id) for doc_id in document_ids)


def decode_document_ids(value: str | None) -> list[str]:
    """Inverse of :func:`encode_document_ids`. Empty/None decodes to ``[]``."""
    if not value:
        return []
    return value.split(",")


def fetch_chat_history_document_ids(db: Session, history_id: int) -> list[str]:
    """Return the document ids stored on a ``ChatHistory`` row.

    Used to thread ``generate_response()``'s doc IDs (persisted when the
    bot's answer was saved) into ``save_feedback()`` when a user reacts to
    that message later. Returns ``[]`` if the row is missing or has none.
    """
    record = db.get(ChatHistory, history_id)
    if record is None:
        return []
    return decode_document_ids(cast(str | None, record.document_ids))


if __name__ == "__main__":
    init_db()
