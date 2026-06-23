import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv
from rag import generate_response
from database import init_db, SessionLocal, ChatHistory
from feedback import save_feedback
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import logging

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "mock_token":
    raise ValueError(
        "TELEGRAM_BOT_TOKEN is not configured. Set a valid bot token in the environment."
    )

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

init_db()

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

        # Generate RAG response. doc_ids is currently unused downstream but kept
        # in the return contract for future feedback-to-document linking.
        try:
            answer, _doc_ids = generate_response(session_id, message.text)
        except Exception as e:
            # Log full detail server-side, but do not leak internal/provider
            # error text back to the user.
            logging.error(f"Error generating response: {e}")
            answer = "Sorry, something went wrong while generating a response. Please try again later."
        
        # Save bot message to DB
        bot_msg = ChatHistory(session_id=session_id, user_id=user_id, message=answer, is_bot=True)
        db.add(bot_msg)
        db.commit()
        
        # Send response with feedback keyboard
        # In a real app we'd need to map bot_msg.id to the inline keyboard
        # For now we use the DB record ID
        await message.answer(answer, reply_markup=get_feedback_keyboard(bot_msg.id))
    finally:
        db.close()

@dp.callback_query(lambda c: c.data and c.data.startswith('fb_'))
async def process_callback_feedback(callback_query: types.CallbackQuery):
    data = callback_query.data.split('_')
    action = data[1] # like or dislike
    chat_id = int(data[2])
    user_id = str(callback_query.from_user.id)
    
    is_positive = (action == "like")
    
    # Save feedback. For simplicity, document_ids is empty here.
    # In a real app, we should retrieve the document_ids associated with chat_id.
    save_feedback(chat_id, user_id, is_positive, correction="", document_ids=[])
    
    await callback_query.answer("Thank you for your feedback!")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
