"""Tests for document-id plumbing between ``ChatHistory`` and ``Feedback``.

Covers the round trip added for issue #9: the doc IDs ``generate_response()``
returns get persisted on the bot's ``ChatHistory`` row when the answer is
saved, then read back and threaded into ``save_feedback()`` instead of being
discarded as ``[]``. Only ``database``/``feedback`` are exercised, both of
which need nothing beyond SQLAlchemy, so the suite still runs with just
``pytest`` and ``sqlalchemy`` installed.
"""
import sys
import types
from unittest.mock import MagicMock

from database import (
    ChatHistory,
    Feedback,
    SessionLocal,
    decode_document_ids,
    encode_document_ids,
    fetch_chat_history_document_ids,
)
from feedback import save_feedback


def _stub_rag(monkeypatch):
    """Replace the heavy ``rag`` module so the negative-feedback branch of
    ``save_feedback`` (which lazily imports ``rag.add_documents``) can run
    without langchain/chromadb installed. Mirrors tests/test_feedback.py.
    """
    fake = types.ModuleType("rag")
    fake.add_documents = MagicMock()
    monkeypatch.setitem(sys.modules, "rag", fake)
    return fake.add_documents


def test_encode_document_ids_joins_with_commas():
    assert encode_document_ids(["doc-1", "doc-2"]) == "doc-1,doc-2"


def test_encode_document_ids_coerces_non_strings():
    assert encode_document_ids([1, 2, 3]) == "1,2,3"


def test_encode_empty_list_is_empty_string():
    assert encode_document_ids([]) == ""


def test_decode_document_ids_splits_on_commas():
    assert decode_document_ids("doc-1,doc-2") == ["doc-1", "doc-2"]


def test_decode_document_ids_handles_empty_and_none():
    assert decode_document_ids("") == []
    assert decode_document_ids(None) == []


def test_fetch_chat_history_document_ids_missing_row_returns_empty():
    db = SessionLocal()
    try:
        assert fetch_chat_history_document_ids(db, 999) == []
    finally:
        db.close()


def test_fetch_chat_history_document_ids_row_without_ids_returns_empty():
    db = SessionLocal()
    try:
        bot_msg = ChatHistory(session_id="chat-1", user_id="user-1", message="hi", is_bot=True)
        db.add(bot_msg)
        db.commit()
        db.refresh(bot_msg)

        assert fetch_chat_history_document_ids(db, bot_msg.id) == []
    finally:
        db.close()


def test_document_ids_round_trip_from_chat_history_into_feedback(monkeypatch):
    """The exact scenario issue #9 describes: doc IDs stored on the bot's
    ChatHistory row (as generate_response() would produce) must be the same
    IDs that end up on the resulting Feedback row, instead of [].
    """
    _stub_rag(monkeypatch)

    db = SessionLocal()
    try:
        bot_msg = ChatHistory(
            session_id="chat-1",
            user_id="user-1",
            message="the answer",
            is_bot=True,
            document_ids=encode_document_ids(["doc-a", "doc-b"]),
        )
        db.add(bot_msg)
        db.commit()
        db.refresh(bot_msg)
        history_id = bot_msg.id

        document_ids = fetch_chat_history_document_ids(db, history_id)
    finally:
        db.close()

    assert document_ids == ["doc-a", "doc-b"]

    save_feedback(
        history_id,
        "user-1",
        is_positive=False,
        correction="the correct answer",
        document_ids=document_ids,
    )

    db = SessionLocal()
    try:
        feedback = db.query(Feedback).one()
    finally:
        db.close()

    assert feedback.chat_id == history_id
    assert feedback.document_ids == "doc-a,doc-b"
    assert feedback.correction == "the correct answer"
