"""Tests for the offline seed/demo script.

Runs :func:`scripts.seed_demo_db.main` against an in-memory database, so it
needs only ``pytest`` + ``sqlalchemy`` -- no Telegram/OpenAI credentials.
"""
from scripts.seed_demo_db import main


def test_seed_demo_populates_rows_and_reports_stats():
    result = main(["--db-url", "sqlite:///:memory:"])

    assert result["chat_history_rows"] == 4
    assert result["feedback"] == {
        "total_feedback": 3,
        "positive": 1,
        "negative": 1,
        "unknown": 1,
    }


def test_seed_demo_stats_partition_the_total():
    result = main(["--db-url", "sqlite:///:memory:"])
    stats = result["feedback"]

    assert stats["positive"] + stats["negative"] + stats["unknown"] == stats["total_feedback"]
