"""Tests for :func:`feedback.save_feedback`.

Only ``feedback`` and ``database`` are imported, both of which need nothing
beyond SQLAlchemy. The ``rag`` module (langchain/chromadb/OpenAI) is stubbed so
the negative-feedback branch can be asserted without heavy dependencies.
"""
import sys
import types
from unittest.mock import MagicMock

from database import Feedback, SessionLocal
from feedback import save_feedback


def _stub_rag(monkeypatch):
    """Replace the heavy ``rag`` module with a lightweight stub.

    ``feedback.process_negative_feedback`` imports ``add_documents`` lazily
    (``from rag import add_documents``), so injecting a fake module into
    ``sys.modules`` lets us observe whether it is called without importing the
    real vector-store stack.
    """
    fake = types.ModuleType("rag")
    fake.add_documents = MagicMock()
    monkeypatch.setitem(sys.modules, "rag", fake)
    return fake.add_documents


def test_save_feedback_happy_path():
    save_feedback(
        chat_id=42,
        user_id="user-1",
        is_positive=True,
        correction="",
        document_ids=[],
    )

    db = SessionLocal()
    try:
        rows = db.query(Feedback).all()
    finally:
        db.close()

    assert len(rows) == 1
    row = rows[0]
    assert row.chat_id == 42
    assert row.user_id == "user-1"
    assert row.is_positive
    assert row.document_ids == ""


def test_document_ids_int_to_str_coercion():
    save_feedback(
        chat_id=1,
        user_id="u",
        is_positive=True,
        correction="",
        document_ids=[1, 2, 3],
    )

    db = SessionLocal()
    try:
        row = db.query(Feedback).one()
    finally:
        db.close()

    assert row.document_ids == "1,2,3"


def test_negative_feedback_without_correction_skips_add_documents(monkeypatch):
    add_documents = _stub_rag(monkeypatch)

    save_feedback(
        chat_id=7,
        user_id="u",
        is_positive=False,
        correction="",
        document_ids=[10],
    )

    add_documents.assert_not_called()

    db = SessionLocal()
    try:
        row = db.query(Feedback).one()
    finally:
        db.close()
    assert not row.is_positive


def test_negative_feedback_with_correction_calls_add_documents(monkeypatch):
    add_documents = _stub_rag(monkeypatch)

    save_feedback(
        chat_id=8,
        user_id="u",
        is_positive=False,
        correction="the sky is blue",
        document_ids=[5],
    )

    add_documents.assert_called_once()
    _, kwargs = add_documents.call_args
    assert "Correction: the sky is blue" in kwargs["texts"]
