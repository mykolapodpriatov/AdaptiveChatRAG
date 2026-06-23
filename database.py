import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    message = Column(Text)
    is_bot = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer) # ID of the bot's message
    user_id = Column(String, index=True)
    is_positive = Column(Boolean) # True for like, False for dislike
    correction = Column(Text, nullable=True) # User's correction text
    document_ids = Column(String) # Comma separated document IDs used for generation
    timestamp = Column(DateTime, default=datetime.utcnow)

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///adaptive_rag.db')
# check_same_thread is a SQLite-only flag; passing it to other backends raises TypeError.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
