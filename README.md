# AdaptiveChatRAG

Self-learning RAG chatbot with memory.
Supports user feedback to correct vector indices and improve search params.

## Features
- **Session-level memory:** Conversation context via LangChain `ConversationBufferMemory`.
- **Retrospective re-indexing:** Adds corrections to ChromaDB **and demotes the
  documents that produced the wrong answer**, so the next retrieval is measurably
  different rather than hoping the correction embeds closer to the query.
- **Real-time evaluation (planned):** Ragas integration for tracking feedback quality.
- **Multi-platform:** Telegram bot (`aiogram`) and FastAPI web panel.

## How feedback changes retrieval

A thumbs-down used to add a correction document and leave the passages that
produced the wrong answer exactly where they were, at the top of the retrieval,
ready to produce it again. `document_ids` was collected, passed all the way
down, and used for nothing.

Now each cited document accumulates a penalty (`document_penalty` table), and
retrieval over-fetches, subtracts each document's penalty from its relevance
score, and then truncates. Over-fetching is what makes this a demotion rather
than a filter: a document pushed down needs somewhere to go, and a document that
should rise has to have been fetched in the first place.

Two properties the penalty has, and why they are not optional:

- **It saturates.** Without a ceiling one determined user could bury a correct
  document permanently. The ceiling (`DEMOTION_CEILING`, default `0.35` on the
  retriever's `[0, 1]` relevance scale) is also deliberately well below what
  would exclude a document, because the UI cannot tell "this document is wrong"
  from "the answer was wrong even though the documents were right". Until it
  can, a thumbs-down is evidence for a demotion, not a veto.
- **It decays**, halving every `PENALTY_HALF_LIFE_DAYS` (default 30), so a
  document does not stay punished for a problem that was fixed months ago.

The penalty is per document, not per (query, document): the Telegram callback
carries the document ids and the message id but not the question text, so a
per-pair penalty is not expressible from what is recorded today. The schema can
grow a nullable query key later without moving any of this.

The arithmetic lives in `ranking.py`, free of the database and of LangChain, so
the shape of the penalty is unit-tested directly.

Tunable with `DEMOTION_CEILING`, `PENALTY_HALF_LIFE_DAYS` and `OVERFETCH_FACTOR`.

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
   export ADMIN_API_KEY="your_admin_api_key"
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

## Offline demo

No Telegram token or OpenAI key? Seed a throwaway SQLite database with a few
chat-history and feedback rows and print the aggregate stats:

```bash
python scripts/seed_demo_db.py
```

By default this writes to `sqlite:///adaptive_rag_demo.db`. Point it elsewhere
(including an ephemeral in-memory database) with `--db-url`:

```bash
python scripts/seed_demo_db.py --db-url "sqlite:///:memory:"
```
