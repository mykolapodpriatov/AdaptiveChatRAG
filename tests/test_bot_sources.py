"""Source-id footer on Telegram replies (issue #12).

CI only installs pytest/sqlalchemy/fastapi/httpx, so aiogram, dotenv, and
rag are stubbed before ``bot`` is imported. ``generate_response`` is then
patched to return a canned ``(answer, doc_ids)`` pair.
"""
import asyncio
import os
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from database import ChatHistory, SessionLocal

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


def _fake_message(text: str = "what is rag?"):
    message = MagicMock()
    message.text = text
    message.chat.id = 101
    message.from_user.id = 202
    message.answer = AsyncMock()
    return message


def test_format_reply_includes_doc_ids():
    from bot import format_reply_with_sources

    reply = format_reply_with_sources("hi", ["doc-1"])
    assert "doc-1" in reply
    assert "Sources:" in reply


def test_format_reply_hides_footer_when_ids_empty():
    from bot import format_reply_with_sources

    assert "Sources:" not in format_reply_with_sources("hi", [])


def test_format_reply_hides_footer_when_ids_unknown():
    from bot import format_reply_with_sources

    assert format_reply_with_sources("hi", ["unknown"]) == "hi"


def test_stubbed_generate_response_puts_doc_id_in_reply(monkeypatch):
    import bot

    monkeypatch.setattr(bot, "generate_response", lambda _sid, _q: ("hi", ["doc-1"]))
    message = _fake_message()
    asyncio.run(bot.handle_message(message))

    message.answer.assert_called_once()
    reply = message.answer.call_args.args[0]
    assert "hi" in reply
    assert "doc-1" in reply
    assert "Sources:" in reply


def test_empty_id_list_does_not_print_sources_line(monkeypatch):
    import bot

    monkeypatch.setattr(bot, "generate_response", lambda _sid, _q: ("hi", []))
    message = _fake_message()
    asyncio.run(bot.handle_message(message))

    reply = message.answer.call_args.args[0]
    assert reply == "hi"
    assert "Sources:" not in reply


def test_stored_history_message_excludes_footer(monkeypatch):
    import bot

    monkeypatch.setattr(bot, "generate_response", lambda _sid, _q: ("hi", ["doc-1"]))
    message = _fake_message()
    asyncio.run(bot.handle_message(message))

    db = SessionLocal()
    try:
        bot_rows = db.query(ChatHistory).filter(ChatHistory.is_bot.is_(True)).all()
    finally:
        db.close()

    assert len(bot_rows) == 1
    assert bot_rows[0].message == "hi"
    assert bot_rows[0].id is not None
    # Keyboard is still keyed off the stored row id.
    keyboard = message.answer.call_args.kwargs.get("reply_markup")
    assert keyboard is not None
