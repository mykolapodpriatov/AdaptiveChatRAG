from database import SessionLocal, Feedback

def save_feedback(chat_id: int, user_id: str, is_positive: bool, correction: str, document_ids: list):
    db = SessionLocal()
    try:
        feedback = Feedback(
            chat_id=chat_id,
            user_id=user_id,
            is_positive=is_positive,
            correction=correction,
            # Coerce ids to str so non-string ids (e.g. ints) don't raise TypeError.
            document_ids=",".join(map(str, document_ids))
        )
        db.add(feedback)
        db.commit()
        
        # Adaptive search update logic
        # If negative feedback, we might want to flag documents or update their weights
        if not is_positive:
            process_negative_feedback(document_ids, correction)
            
    finally:
        db.close()

def process_negative_feedback(document_ids: list, correction: str):
    # This is a placeholder for retrospective re-indexing or corrective RAG
    # In a full implementation, we might lower the weight of these documents
    # or add the user's correction to the vector store
    
    # E.g., if there's a correction, index it
    if correction:
        from rag import add_documents
        add_documents(
            texts=[f"Correction: {correction}"],
            metadatas=[{"source": "user_feedback", "type": "correction"}]
        )
