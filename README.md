# AdaptiveChatRAG

Self-learning RAG chatbot with memory.
Supports user feedback to correct vector indices and improve search params.

## Features
- **Session-level memory:** Conversation context via LangChain `ConversationBufferMemory`.
- **Retrospective re-indexing:** Adds corrections to ChromaDB.
- **Real-time evaluation (planned):** Ragas integration for tracking feedback quality.
- **Multi-platform:** Telegram bot (`aiogram`) and FastAPI web panel.

## Installation

1. Clone repo
2. Create virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```bash
   export TELEGRAM_BOT_TOKEN="your_token"
   export OPENAI_API_KEY="your_openai_api_key"
   ```
4. Initialize the database:
   ```bash
   python database.py
   ```
5. Run Bot:
   ```bash
   python bot.py
   ```
6. Run API:
   ```bash
   uvicorn app:app --reload
   ```
