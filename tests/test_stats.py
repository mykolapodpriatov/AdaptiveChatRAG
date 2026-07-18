"""Tests for :func:`database.compute_feedback_stats` and
:func:`database.fetch_session_history`.

These helpers were extracted out of the FastAPI layer precisely so they can be
exercised with only ``pytest`` + ``sqlalchemy`` installed -- no FastAPI import.
"""
from datetime import datetime, timedelta

from database import (
    ChatHistory,
    Feedback,
    SessionLocal,
    compute_feedback_stats,
    fetch_session_history,
)


def _add_feedback(db, is_positive):
    db.add(
        Feedback(
            chat_id=1,
            user_id="u",
            is_positive=is_positive,
            correction="",
            document_ids="",
        )
    )


def test_compute_feedback_stats_buckets_true_false_and_null():
    db = SessionLocal()
    try:
        _add_feedback(db, True)
        _add_feedback(db, True)
        _add_feedback(db, False)
        _add_feedback(db, None)  # NULL must NOT be counted as negative.
        db.commit()

        stats = compute_feedback_stats(db)
    finally:
        db.close()

    assert stats == {
        "total_feedback": 4,
        "positive": 2,
        "negative": 1,
        "unknown": 1,
    }
    # Invariant: the three buckets always partition the total.
    assert stats["positive"] + stats["negative"] + stats["unknown"] == stats["total_feedback"]


def test_compute_feedback_stats_empty_db():
    db = SessionLocal()
    try:
        stats = compute_feedback_stats(db)
    finally:
        db.close()

    assert stats == {
        "total_feedback": 0,
        "positive": 0,
        "negative": 0,
        "unknown": 0,
    }


def test_fetch_session_history_filters_by_session_and_orders_by_time():
    base = datetime(2024, 1, 1, 12, 0, 0)
    db = SessionLocal()
    try:
        db.add_all(
            [
                ChatHistory(
                    session_id="s1",
                    user_id="u",
                    message="second",
                    is_bot=True,
                    timestamp=base + timedelta(seconds=1),
                ),
                ChatHistory(
                    session_id="s1",
                    user_id="u",
                    message="first",
                    is_bot=False,
                    timestamp=base,
                ),
                ChatHistory(
                    session_id="other",
                    user_id="u",
                    message="ignored",
                    is_bot=False,
                    timestamp=base,
                ),
            ]
        )
        db.commit()

        rows = fetch_session_history(db, "s1")
    finally:
        db.close()

    assert [row.message for row in rows] == ["first", "second"]


def test_fetch_session_history_unknown_session_is_empty():
    db = SessionLocal()
    try:
        rows = fetch_session_history(db, "does-not-exist")
    finally:
        db.close()

    assert rows == []
