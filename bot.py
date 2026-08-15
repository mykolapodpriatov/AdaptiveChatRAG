import asyncio
import os
from typing import cast

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from rag import generate_response
from database import (
    ChatHistory,
    SessionLocal,
    encode_document_ids,
    fetch_chat_history_document_ids,
    init_db,
)
from feedback import FeedbackCallbackError, parse_feedback_callback, save_feedback
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import logging

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "mock_token":
    raise ValueError(
        "TELEGRAM_BOT_TOKEN is not configured. Set a valid bot token in the environment."
    )

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

init_db()


class FeedbackStates(StatesGroup):
    # Entered right after a 👎 press; the next text message from that user
    # is captured as the correction instead of being treated as a new
    # question. Cleared as soon as that message is handled.
    awaiting_correction = State()


def format_reply_with_sources(answer: str, document_ids: list) -> str:
    """Append a short ``Sources:`` footer for the Telegram reply.

    Hidden when there are no ids, or when every id is the ``unknown``
    sentinel ``generate_response()`` uses for sources with no metadata id.
    The stored ``ChatHistory.message`` stays the raw answer; only the
    outgoing Telegram text includes this footer.
    """
    visible = [
        str(doc_id)
        for doc_id in document_ids
        if str(doc_id) and str(doc_id) != "unknown"
    ]
    if not visible:
        return answer
    return f"{answer}\n\nSources: {', '.join(visible)}"


def get_feedback_keyboard(history_id: int):
    # history_id is the ChatHistory primary key of the bot message, embedded in
    # callback data so feedback can be linked back to the stored message.
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="👍", callback_data=f"fb_like_{history_id}"),
            InlineKeyboardButton(text="👎", callback_data=f"fb_dislike_{history_id}")
        ]
    ])
    return keyboard

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello! I am AdaptiveChatRAG. Ask me anything!")

@dp.message(FeedbackStates.awaiting_correction)
async def handle_feedback_correction(message: types.Message, state: FSMContext):
    # Registered before the catch-all handle_message() below so it wins
    # while the user is in this state: the message is the correction text
    # for a 👎, not a new question.
    data = await state.get_data()
    await state.clear()

    user_id = str(message.from_user.id)
    correction = message.text or ""
    save_feedback(
        cast(int, data["history_id"]),
        user_id,
        is_positive=False,
        correction=correction,
        document_ids=data.get("document_ids", []),
    )

    if correction:
        await message.answer("Thanks for the correction!")
    else:
        await message.answer("Thanks — feedback recorded without a correction.")

@dp.message()
async def handle_message(message: types.Message):
    # Non-text updates (photos, stickers, joins, etc.) have message.text == None,
    # which would crash RAG/DB downstream. Reject them early.
    if not message.text:
        await message.answer("Sorry, I can only process text messages right now.")
        return

    session_id = str(message.chat.id)
    user_id = str(message.from_user.id)

    # Save user message to DB
    db = SessionLocal()
    try:
        user_msg = ChatHistory(session_id=session_id, user_id=user_id, message=message.text, is_bot=False)
        db.add(user_msg)
        db.commit()

        # Generate RAG response. doc_ids is persisted on the bot's ChatHistory
        # row below so a later 👎 on this message can thread the real IDs
        # into save_feedback() instead of an empty list.
        doc_ids: list = []
        try:
            answer, doc_ids = generate_response(session_id, message.text)
        except Exception as e:
            # Log full detail server-side, but do not leak internal/provider
            # error text back to the user.
            logging.error(f"Error generating response: {e}")
            answer = "Sorry, something went wrong while generating a response. Please try again later."

        # Save bot message to DB
        bot_msg = ChatHistory(
            session_id=session_id,
            user_id=user_id,
            message=answer,
            is_bot=True,
            document_ids=encode_document_ids(doc_ids),
        )
        db.add(bot_msg)
        db.commit()

        # Send response with feedback keyboard. Footer is display-only so
        # ChatHistory.message stays the raw answer; feedback still keys
        # off bot_msg.id.
        await message.answer(
            format_reply_with_sources(answer, doc_ids),
            reply_markup=get_feedback_keyboard(cast(int, bot_msg.id)),
        )
    finally:
        db.close()

@dp.callback_query(lambda c: c.data and c.data.startswith('fb_'))
async def process_callback_feedback(callback_query: types.CallbackQuery, state: FSMContext):
    # Never trust callback_data: it can be stale, truncated, or spoofed. Parse
    # defensively and fail with a friendly message instead of crashing.
    try:
        action, history_id = parse_feedback_callback(callback_query.data)
    except FeedbackCallbackError as e:
        logging.warning(f"Ignoring malformed feedback callback: {e}")
        await callback_query.answer("Sorry, that feedback button is no longer valid.")
        return

    user_id = str(callback_query.from_user.id)
    is_positive = (action == "like")

    # history_id is the ChatHistory row of the bot's message; recover the
    # real document IDs generate_response() used instead of passing [].
    db = SessionLocal()
    try:
        document_ids = fetch_chat_history_document_ids(db, history_id)
    finally:
        db.close()

    if is_positive:
        save_feedback(history_id, user_id, is_positive=True, correction="", document_ids=document_ids)
        await callback_query.answer("Thank you for your feedback!")
        return

    # 👎: hold off on saving until the user has a chance to supply the
    # correct answer, so process_negative_feedback() gets real correction
    # text to index instead of an empty string.
    await state.set_state(FeedbackStates.awaiting_correction)
    await state.update_data(history_id=history_id, document_ids=document_ids)
    await callback_query.answer()
    if callback_query.message is not None:
        await callback_query.message.answer(
            "What should the answer have been? Reply with the correct answer."
        )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
