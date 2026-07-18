from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from database import SessionLocal, compute_feedback_stats, fetch_session_history

app = FastAPI(title="AdaptiveChatRAG API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"message": "Welcome to AdaptiveChatRAG Admin Panel API"}


@app.get("/stats/feedback")
def get_feedback_stats(db: Session = Depends(get_db)):
    return compute_feedback_stats(db)


@app.get("/history/{session_id}")
def get_history(session_id: str, db: Session = Depends(get_db)):
    return fetch_session_history(db, session_id)
