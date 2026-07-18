"""Offline seed/demo script for the feedback database.

Populate a throwaway SQLite database with a handful of chat-history and
feedback rows and print the aggregate stats -- no Telegram token or OpenAI key
required. This is the fastest way to eyeball the schema and the
``compute_feedback_stats`` buckets.

Usage::

    python scripts/seed_demo_db.py
    python scripts/seed_demo_db.py --db-url sqlite:///scratch.db

Only ``database`` (SQLAlchemy) is imported, so it runs in the same minimal
environment as the test suite.
"""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python scripts/seed_demo_db.py`` from the repo root: in that mode the
# script's own directory -- not the project root -- lands on sys.path, so add
# the parent directory to make ``database`` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import (  # noqa: E402  -- import must follow the sys.path bootstrap above
    ChatHistory,
    Feedback,
    SessionLocal,
    compute_feedback_stats,
    init_db,
    init_engine,
)
from sqlalchemy.orm import Session  # noqa: E402

DEFAULT_DB_URL = "sqlite:///adaptive_rag_demo.db"


def _seed_rows(db: Session) -> None:
    """Insert a small, deterministic demo dataset."""
    db.add_all(
        [
            ChatHistory(session_id="demo-1", user_id="alice", message="What is RAG?", is_bot=False),
            ChatHistory(session_id="demo-1", user_id="alice", message="Retrieval-Augmented Generation.", is_bot=True),
            ChatHistory(session_id="demo-2", user_id="bob", message="Ping?", is_bot=False),
            ChatHistory(session_id="demo-2", user_id="bob", message="Pong!", is_bot=True),
        ]
    )
    db.add_all(
        [
            Feedback(chat_id=2, user_id="alice", is_positive=True, correction="", document_ids="doc-1"),
            Feedback(chat_id=4, user_id="bob", is_positive=False, correction="Too terse", document_ids="doc-2"),
            # A NULL vote exercises the explicit "unknown" bucket.
            Feedback(chat_id=4, user_id="carol", is_positive=None, correction="", document_ids=""),
        ]
    )
    db.commit()


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Seed the demo database and print aggregate stats.

    Returns a dict with the chat-history row count and the feedback stats so
    callers (including tests) can assert on the result without parsing stdout.
    """
    parser = argparse.ArgumentParser(
        description="Seed a demo feedback database and print aggregate stats."
    )
    parser.add_argument(
        "--db-url",
        default=DEFAULT_DB_URL,
        help="SQLAlchemy database URL (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    # Bind the engine to the requested URL first, then create the schema.
    init_engine(args.db_url)
    init_db(args.db_url)

    db = SessionLocal()
    try:
        _seed_rows(db)
        history_count = db.query(ChatHistory).count()
        stats = compute_feedback_stats(db)
    finally:
        db.close()

    print(f"Seeded demo database at {args.db_url}")
    print(f"  chat_history rows: {history_count}")
    print(
        "  feedback: "
        f"total={stats['total_feedback']} "
        f"positive={stats['positive']} "
        f"negative={stats['negative']} "
        f"unknown={stats['unknown']}"
    )

    return {"chat_history_rows": history_count, "feedback": stats}


if __name__ == "__main__":
    main()
