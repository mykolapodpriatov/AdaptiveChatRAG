import csv
import io
import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from sqlalchemy.orm import Session

from database import (
    SessionLocal,
    compute_feedback_stats,
    fetch_feedback_rows,
    fetch_session_history,
)

_FEEDBACK_CSV_HEADER = [
    "history_id",
    "user_id",
    "is_positive",
    "correction",
    "document_ids",
    "created_at",
]

app = FastAPI(title="AdaptiveChatRAG API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_admin_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Reject the request unless ``X-API-Key`` matches ``ADMIN_API_KEY``.

    Applied to admin-only routes. ``/`` stays open as an unauthenticated
    health check. An unset ``ADMIN_API_KEY`` fails closed (401) rather than
    leaving the admin panel world-readable.
    """
    expected = os.getenv("ADMIN_API_KEY") or ""
    provided = x_api_key or ""
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@app.get("/")
def read_root():
    return {"message": "Welcome to AdaptiveChatRAG Admin Panel API"}


@app.get("/stats/feedback", dependencies=[Depends(require_admin_api_key)])
def get_feedback_stats(db: Session = Depends(get_db)):
    return compute_feedback_stats(db)


@app.get("/history/{session_id}", dependencies=[Depends(require_admin_api_key)])
def get_history(session_id: str, db: Session = Depends(get_db)):
    return fetch_session_history(db, session_id)


def _feedback_csv(rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_FEEDBACK_CSV_HEADER)
    for row in rows:
        created_at = row.timestamp.isoformat() if row.timestamp else ""
        writer.writerow(
            [
                row.chat_id,
                row.user_id,
                row.is_positive,
                row.correction or "",
                row.document_ids or "",
                created_at,
            ]
        )
    return buf.getvalue()


@app.get("/feedback.csv", dependencies=[Depends(require_admin_api_key)])
def export_feedback_csv(db: Session = Depends(get_db)):
    body = _feedback_csv(fetch_feedback_rows(db))
    return Response(content=body, media_type="text/csv")
