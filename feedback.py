from database import (
    Feedback,
    SessionLocal,
    encode_document_ids,
    record_negative_documents,
)

# Callback payloads are formatted as "fb_<action>_<history_id>" (see bot.py).
_CALLBACK_PREFIX = "fb"
_VALID_ACTIONS = ("like", "dislike")


class FeedbackCallbackError(ValueError):
    """Raised when a Telegram feedback callback payload is malformed."""


def parse_feedback_callback(data: str) -> tuple[str, int]:
    """Parse a ``fb_<action>_<history_id>`` callback payload.

    Returns ``(action, history_id)`` where ``action`` is ``"like"`` or
    ``"dislike"``. Raises :class:`FeedbackCallbackError` for any malformed
    payload (wrong prefix, missing/unknown action, missing or non-numeric id,
    or stray extra segments) so callers never hit an unguarded ``IndexError``
    or ``ValueError``.

    Importable with no environment configured: it touches neither the database
    nor any network client.
    """
    if not isinstance(data, str):
        raise FeedbackCallbackError(f"Feedback callback must be a string, got {type(data)!r}")

    # Split into at most 3 parts; the id segment must then be a bare integer,
    # which rejects trailing junk such as "fb_like_1_2".
    parts = data.split("_", 2)
    if len(parts) != 3 or parts[0] != _CALLBACK_PREFIX:
        raise FeedbackCallbackError(f"Malformed feedback callback: {data!r}")

    _, action, raw_id = parts
    if action not in _VALID_ACTIONS:
        raise FeedbackCallbackError(f"Unknown feedback action {action!r} in {data!r}")
    if not raw_id.isdigit():
        raise FeedbackCallbackError(f"Non-numeric feedback id {raw_id!r} in {data!r}")

    return action, int(raw_id)


def save_feedback(chat_id: int, user_id: str, is_positive: bool, correction: str, document_ids: list):
    db = SessionLocal()
    try:
        feedback = Feedback(
            chat_id=chat_id,
            user_id=user_id,
            is_positive=is_positive,
            correction=correction,
            # Coerce ids to str so non-string ids (e.g. ints) don't raise TypeError.
            document_ids=encode_document_ids(document_ids)
        )
        db.add(feedback)
        db.commit()
        
        # Adaptive search update logic.
        if not is_positive:
            # Record the demotion inside this session, before the correction is
            # indexed: the correction is a best-effort side effect that reaches
            # out to the vector store, and a failure there must not lose the
            # thing we can record reliably.
            record_negative_documents(db, document_ids)
            process_negative_feedback(document_ids, correction)
            
    finally:
        db.close()

def process_negative_feedback(document_ids: list, correction: str):
    """Index the user's correction.

    Demoting the documents that produced the wrong answer happens in
    :func:`save_feedback` via :func:`database.record_negative_documents`, and
    the demotion is applied at retrieval time by :mod:`rag`. This function is
    only the indexing half.
    """
    if correction:
        from rag import add_documents
        add_documents(
            texts=[f"Correction: {correction}"],
            metadatas=[{"source": "user_feedback", "type": "correction"}]
        )
