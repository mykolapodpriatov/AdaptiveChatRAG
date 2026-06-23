from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, Feedback, ChatHistory

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
    total = db.query(Feedback).count()
    positive = db.query(Feedback).filter(Feedback.is_positive).count()
    negative = total - positive
    return {"total_feedback": total, "positive": positive, "negative": negative}

@app.get("/history/{session_id}")
def get_history(session_id: str, db: Session = Depends(get_db)):
    history = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp).all()
    return history
