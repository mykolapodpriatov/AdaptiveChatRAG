"""Tests for the aiogram feedback handlers themselves (issue #20).

``tests/test_callback_parsing.py`` only exercises the pure
``parse_feedback_callback`` parser, and ``tests/test_feedback.py`` only
exercises ``save_feedback`` directly. Neither drives
``process_callback_feedback`` or ``handle_feedback_correction``, the two
handlers that actually compose the parser, ``fetch_chat_history_document_ids``,
FSM state, and ``save_feedback`` into the real 👍/👎 flow. This module drives
them the way aiogram would, through fake ``CallbackQuery``/``Message``/
``FSMContext`` objects, so a regression in that wiring (e.g. document_ids
dropped between the callback and the correction) fails a test instead of only
ever showing up against a live bot.

CI only installs pytest/sqlalchemy/fastapi/httpx, so aiogram, dotenv, and rag
are stubbed before ``bot`` is imported. Mirrors tests/test_bot_sources.py.
"""
import asyncio
import os
import sys
import types
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from database import ChatHistory, Feedback, SessionLocal, encode_document_ids

# A non-mock token so bot.py's import-time guard does not raise.
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token-not-mock")


def _passthrough(*_args, **_kwargs):
    def deco(fn):
        return fn

    return deco


def _stub_module(name: str, **attrs: object) -> Any:
    """Build a throwaway module carrying ``attrs``.

    Returns ``Any`` because mypy rejects attribute assignment on
    ``types.ModuleType``: these names exist only for this test run.
    """
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


class _FakeDispatcher:
    def __init__(self, *args, **kwargs):
        pass

    message = staticmethod(_passthrough)
    callback_query = staticmethod(_passthrough)


class _State:
    pass


class _StatesGroup:
    pass


def _install_bot_stubs() -> None:
    if getattr(sys.modules.get("aiogram"), "_is_test_stub", False):
        return

    aiogram_types = _stub_module(
        "aiogram.types",
        Message=type("Message", (), {}),
        CallbackQuery=type("CallbackQuery", (), {}),
        InlineKeyboardMarkup=MagicMock(),
        InlineKeyboardButton=MagicMock(),
    )
    aiogram_filters = _stub_module("aiogram.filters", Command=MagicMock())
    aiogram_fsm_context = _stub_module("aiogram.fsm.context", FSMContext=type("FSMContext", (), {}))
    aiogram_fsm_state = _stub_module("aiogram.fsm.state", State=_State, StatesGroup=_StatesGroup)
    aiogram_fsm_storage = _stub_module("aiogram.fsm.storage")
    aiogram_fsm_memory = _stub_module("aiogram.fsm.storage.memory", MemoryStorage=MagicMock())
    aiogram_fsm = _stub_module("aiogram.fsm")

    aiogram = _stub_module(
        "aiogram",
        _is_test_stub=True,
        Bot=MagicMock(),
        Dispatcher=_FakeDispatcher,
        types=aiogram_types,
        filters=aiogram_filters,
        fsm=aiogram_fsm,
    )

    dotenv = _stub_module("dotenv", load_dotenv=lambda: None)
    rag = _stub_module("rag", generate_response=MagicMock(return_value=("hi", ["doc-1"])))

    sys.modules.update(
        {
            "aiogram": aiogram,
            "aiogram.types": aiogram_types,
            "aiogram.filters": aiogram_filters,
            "aiogram.fsm": aiogram_fsm,
            "aiogram.fsm.context": aiogram_fsm_context,
            "aiogram.fsm.state": aiogram_fsm_state,
            "aiogram.fsm.storage": aiogram_fsm_storage,
            "aiogram.fsm.storage.memory": aiogram_fsm_memory,
            "dotenv": dotenv,
            "rag": rag,
        }
    )

_install_bot_stubs()


def _stub_rag(monkeypatch):
    """Replace the shared ``rag`` stub with one that also has ``add_documents``.

    ``feedback.process_negative_feedback`` imports ``add_documents`` lazily
    (``from rag import add_documents``) only when a correction is non-empty.
    The collection-time stub installed by ``_install_bot_stubs`` (here or in
    tests/test_bot_sources.py, whichever import wins) only defines
    ``generate_response``, so tests that reach that branch need this. Mirrors
    tests/test_feedback.py.
    """
    fake = types.ModuleType("rag")
    fake.add_documents = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rag", fake)
    return fake.add_documents


def _fake_message(text: str = "the correct answer"):
    message = MagicMock()
    message.text = text
    message.from_user.id = 202
    message.answer = AsyncMock()
    return message


def _fake_callback_query(data: str, message=None):
    callback_query = MagicMock()
    callback_query.data = data
    callback_query.from_user.id = 303
    callback_query.answer = AsyncMock()
    callback_query.message = message
    return callback_query


class _FakeFSMContext:
    """Minimal stand-in for aiogram's ``FSMContext``.

    Backed by a plain dict instead of real storage, which is enough to assert
    what ``process_callback_feedback`` writes and what
    ``handle_feedback_correction`` later reads back.
    """

    def __init__(self) -> None:
        self.state: object = None
        self.data: dict = {}

    async def set_state(self, state: object) -> None:
        self.state = state

    async def update_data(self, **kwargs: object) -> None:
        self.data.update(kwargs)

    async def get_data(self) -> dict:
        return dict(self.data)

    async def clear(self) -> None:
        self.state = None
        self.data = {}


def _make_bot_history_row(document_ids: list) -> int:
    """Persist a bot ``ChatHistory`` row carrying ``document_ids``, return its id.

    Mirrors what ``handle_message`` stores after a real answer, so
    ``fetch_chat_history_document_ids`` (which ``process_callback_feedback``
    calls) has something real to read back.
    """
    db = SessionLocal()
    try:
        row = ChatHistory(
            session_id="chat-1",
            user_id="user-1",
            message="the answer",
            is_bot=True,
            document_ids=encode_document_ids(document_ids),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return cast(int, row.id)
    finally:
        db.close()


def test_like_callback_saves_feedback_immediately():
    import bot

    history_id = _make_bot_history_row(["doc-1", "doc-2"])
    callback_query = _fake_callback_query(f"fb_like_{history_id}")
    state = _FakeFSMContext()

    asyncio.run(bot.process_callback_feedback(callback_query, state))

    db = SessionLocal()
    try:
        row = db.query(Feedback).one()
    finally:
        db.close()

    assert row.chat_id == history_id
    assert row.is_positive is True
    assert row.document_ids == "doc-1,doc-2"
    callback_query.answer.assert_awaited_once_with("Thank you for your feedback!")
    # A 👍 never enters the correction flow.
    assert state.state is None


def test_dislike_callback_defers_save_and_stores_state():
    import bot

    history_id = _make_bot_history_row(["doc-3"])
    followup = _fake_message()
    callback_query = _fake_callback_query(f"fb_dislike_{history_id}", message=followup)
    state = _FakeFSMContext()

    asyncio.run(bot.process_callback_feedback(callback_query, state))

    db = SessionLocal()
    try:
        rows = db.query(Feedback).all()
    finally:
        db.close()

    assert rows == []  # save_feedback must wait for handle_feedback_correction.
    assert state.state is bot.FeedbackStates.awaiting_correction
    assert state.data == {"history_id": history_id, "document_ids": ["doc-3"]}
    callback_query.answer.assert_awaited_once_with()
    followup.answer.assert_awaited_once_with(
        "What should the answer have been? Reply with the correct answer."
    )


def test_malformed_callback_is_ignored():
    import bot

    callback_query = _fake_callback_query("fb_love_1")
    state = _FakeFSMContext()

    asyncio.run(bot.process_callback_feedback(callback_query, state))

    db = SessionLocal()
    try:
        rows = db.query(Feedback).all()
    finally:
        db.close()

    assert rows == []
    assert state.state is None
    callback_query.answer.assert_awaited_once_with(
        "Sorry, that feedback button is no longer valid."
    )


def test_correction_with_text_saves_negative_feedback_and_thanks_user(monkeypatch):
    import bot

    _stub_rag(monkeypatch)

    state = _FakeFSMContext()
    state.data = {"history_id": 55, "document_ids": ["doc-9"]}
    message = _fake_message("the sky is blue")

    asyncio.run(bot.handle_feedback_correction(message, state))

    db = SessionLocal()
    try:
        row = db.query(Feedback).one()
    finally:
        db.close()

    assert row.chat_id == 55
    assert row.is_positive is False
    assert row.correction == "the sky is blue"
    assert row.document_ids == "doc-9"
    message.answer.assert_awaited_once_with("Thanks for the correction!")
    # State is cleared so a later message is treated as a new question.
    assert state.state is None
    assert state.data == {}


def test_correction_with_empty_text_still_saves_feedback():
    import bot

    state = _FakeFSMContext()
    state.data = {"history_id": 56, "document_ids": []}
    message = _fake_message("")

    asyncio.run(bot.handle_feedback_correction(message, state))

    db = SessionLocal()
    try:
        row = db.query(Feedback).one()
    finally:
        db.close()

    assert row.chat_id == 56
    assert row.is_positive is False
    assert row.correction == ""
    message.answer.assert_awaited_once_with(
        "Thanks — feedback recorded without a correction."
    )


def test_dislike_then_correction_round_trip_preserves_document_ids(monkeypatch):
    """The exact wiring issue #9 and issue #18's demotion depend on: the ids
    generate_response() stored on the bot's ChatHistory row must survive a
    👎 plus a correction and land on the resulting Feedback row unchanged.
    """
    import bot

    _stub_rag(monkeypatch)

    history_id = _make_bot_history_row(["doc-a", "doc-b"])
    dislike_query = _fake_callback_query(f"fb_dislike_{history_id}", message=_fake_message())
    state = _FakeFSMContext()
    asyncio.run(bot.process_callback_feedback(dislike_query, state))

    correction_message = _fake_message("the correct answer")
    asyncio.run(bot.handle_feedback_correction(correction_message, state))

    db = SessionLocal()
    try:
        row = db.query(Feedback).one()
    finally:
        db.close()

    assert row.chat_id == history_id
    assert row.document_ids == "doc-a,doc-b"
    assert row.correction == "the correct answer"
