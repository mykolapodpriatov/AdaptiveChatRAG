"""Tests for recording negative feedback against documents.

Only ``database`` and ``feedback`` are imported, both of which need nothing
beyond SQLAlchemy; the heavy ``rag`` module is stubbed the same way
``test_feedback.py`` does it.
"""

import sys
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from database import (
    DocumentPenalty,
    SessionLocal,
    fetch_document_penalties,
    record_negative_documents,
)
from feedback import save_feedback

NOW = datetime(2026, 6, 1, 12, 0)


def _stub_rag(monkeypatch):
    fake = types.ModuleType("rag")
    fake.add_documents = MagicMock()
    monkeypatch.setitem(sys.modules, "rag", fake)
    return fake.add_documents


def _penalties() -> dict[str, DocumentPenalty]:
    db = SessionLocal()
    try:
        return {row.document_id: row for row in fetch_document_penalties(db)}
    finally:
        db.close()


def test_a_thumbs_down_records_a_penalty_for_every_cited_document(monkeypatch):
    _stub_rag(monkeypatch)

    save_feedback(
        chat_id=1,
        user_id="u",
        is_positive=False,
        correction="",
        document_ids=["doc-a", "doc-b"],
    )

    rows = _penalties()
    assert set(rows) == {"doc-a", "doc-b"}
    assert rows["doc-a"].negative_count == 1


def test_a_thumbs_up_records_nothing(monkeypatch):
    _stub_rag(monkeypatch)

    save_feedback(chat_id=1, user_id="u", is_positive=True, correction="", document_ids=["doc-a"])

    assert _penalties() == {}


def test_repeated_thumbs_down_accumulates(monkeypatch):
    _stub_rag(monkeypatch)

    for _ in range(3):
        save_feedback(
            chat_id=1, user_id="u", is_positive=False, correction="", document_ids=["doc-a"]
        )

    assert _penalties()["doc-a"].negative_count == 3


def test_one_answer_citing_a_document_twice_counts_once():
    # One thumbs-down is one piece of evidence per document.
    db = SessionLocal()
    try:
        touched = record_negative_documents(db, ["doc-a", "doc-a"], now=NOW)
    finally:
        db.close()

    assert touched == 1
    assert _penalties()["doc-a"].negative_count == 1


def test_the_unknown_placeholder_is_not_recorded():
    # rag.generate_response uses 'unknown' for a document with no id; a penalty
    # on it would match nothing and accumulate forever.
    db = SessionLocal()
    try:
        touched = record_negative_documents(db, ["unknown", "", "  ", "real"], now=NOW)
    finally:
        db.close()

    assert touched == 1
    assert set(_penalties()) == {"real"}


def test_feedback_with_no_document_ids_changes_nothing_and_does_not_raise(monkeypatch):
    # This is what the current callback path can produce.
    _stub_rag(monkeypatch)

    save_feedback(chat_id=1, user_id="u", is_positive=False, correction="", document_ids=[])

    assert _penalties() == {}


def test_the_timestamp_is_the_most_recent_thumbs_down():
    db = SessionLocal()
    try:
        record_negative_documents(db, ["doc-a"], now=NOW - timedelta(days=10))
        record_negative_documents(db, ["doc-a"], now=NOW)
    finally:
        db.close()

    row = _penalties()["doc-a"]
    assert row.negative_count == 2
    assert row.last_negative_at == NOW


def test_fetching_can_be_restricted_to_a_set_of_ids():
    db = SessionLocal()
    try:
        record_negative_documents(db, ["a", "b", "c"], now=NOW)
        subset = fetch_document_penalties(db, ["a", "c"])
        empty = fetch_document_penalties(db, [])
    finally:
        db.close()

    assert {row.document_id for row in subset} == {"a", "c"}
    assert empty == []


def test_the_demotion_is_recorded_even_when_indexing_the_correction_fails(monkeypatch):
    """Git-style ordering: record what we can record reliably first.

    The correction reaches out to the vector store and may fail; the penalty is
    a local write that should not be lost with it.
    """
    fake = types.ModuleType("rag")
    fake.add_documents = MagicMock(side_effect=RuntimeError("vector store down"))
    monkeypatch.setitem(sys.modules, "rag", fake)

    try:
        save_feedback(
            chat_id=1,
            user_id="u",
            is_positive=False,
            correction="the real answer",
            document_ids=["doc-a"],
        )
    except RuntimeError:
        pass

    assert _penalties()["doc-a"].negative_count == 1
