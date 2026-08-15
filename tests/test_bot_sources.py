"""Source-id footer on Telegram replies (issue #12).

CI only installs pytest/sqlalchemy/fastapi/httpx, so aiogram, dotenv, and
rag are stubbed before ``bot`` is imported. ``generate_response`` is then
patched to return a canned ``(answer, doc_ids)`` pair.
"""
import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

from database import ChatHistory, SessionLocal

# A non-mock token so bot.py's import-time guard does not raise.
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token-not-mock")


def _passthrough(*_args, **_kwargs):
    def deco(fn):
        return fn

    return deco


class _FakeDispatcher:
    def __init__(self, *args, **kwargs):
        pass

    message = staticmethod(_passthrough)
    callback_query = staticmethod(_passthrough)


def _install_bot_stubs() -> None:
    if getattr(sys.modules.get("aiogram"), "_is_test_stub", False):
        return

    aiogram = types.ModuleType("aiogram")
    aiogram._is_test_stub = True
    aiogram.Bot = MagicMock()
    aiogram.Dispatcher = _FakeDispatcher

    aiogram_types = types.ModuleType("aiogram.types")
    aiogram_types.Message = type("Message", (), {})
    aiogram_types.CallbackQuery = type("CallbackQuery", (), {})
    aiogram_types.InlineKeyboardMarkup = MagicMock()
    aiogram_types.InlineKeyboardButton = MagicMock()

    aiogram_filters = types.ModuleType("aiogram.filters")
    aiogram_filters.Command = MagicMock()

    aiogram_fsm = types.ModuleType("aiogram.fsm")
    aiogram_fsm_context = types.ModuleType("aiogram.fsm.context")
    aiogram_fsm_context.FSMContext = type("FSMContext", (), {})
    aiogram_fsm_state = types.ModuleType("aiogram.fsm.state")

    class State:
        pass

    class StatesGroup:
        pass

    aiogram_fsm_state.State = State
    aiogram_fsm_state.StatesGroup = StatesGroup
    aiogram_fsm_storage = types.ModuleType("aiogram.fsm.storage")
    aiogram_fsm_memory = types.ModuleType("aiogram.fsm.storage.memory")
    aiogram_fsm_memory.MemoryStorage = MagicMock()

    aiogram.types = aiogram_types
    aiogram.filters = aiogram_filters
    aiogram.fsm = aiogram_fsm

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None

    rag = types.ModuleType("rag")
    rag.generate_response = MagicMock(return_value=("hi", ["doc-1"]))

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
