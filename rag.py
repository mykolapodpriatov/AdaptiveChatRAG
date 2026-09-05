import os
from collections import OrderedDict
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv

load_dotenv()

# We will use Chroma as vector store. Embeddings/vectorstore are initialized
# lazily so importing this module does not crash when OPENAI_API_KEY is unset.
_embeddings = None
_vectorstore = None

def get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = OpenAIEmbeddings()
        _vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=_embeddings)
    return _vectorstore

def add_documents(texts, metadatas=None):
    # Chroma 0.4.x+ persists automatically; persist() is deprecated/a no-op.
    get_vectorstore().add_texts(texts=texts, metadatas=metadatas)

DEFAULT_K = int(os.getenv("RETRIEVER_K", "4"))


def _document_id(document) -> str:
    """The id a piece of feedback would name this document by."""
    return str(document.metadata.get("id", "unknown"))


def retrieve_with_demotion(question: str, k: int = DEFAULT_K, now=None):
    """Retrieve for ``question``, demoting documents with negative feedback.

    Over-fetches, subtracts each document's accumulated penalty from its
    relevance score, then truncates. Over-fetching is what makes this a
    demotion rather than a filter: a document pushed down needs somewhere to go,
    and a document that should rise has to have been fetched in the first place.

    The penalty is bounded and decays; see :mod:`ranking` for why.
    """
    from database import SessionLocal, fetch_document_penalties
    from ranking import PenaltyRecord, demote, overfetch_size, penalties_by_id

    pairs = get_vectorstore().similarity_search_with_relevance_scores(
        question, k=overfetch_size(k)
    )
    documents = {_document_id(document): document for document, _score in pairs}
    candidates = [(_document_id(document), float(score)) for document, score in pairs]

    db = SessionLocal()
    try:
        rows = fetch_document_penalties(db, list(documents))
        penalties = penalties_by_id(
            PenaltyRecord(
                document_id=row.document_id,
                negative_count=row.negative_count or 0,
                last_negative_at=row.last_negative_at,
            )
            for row in rows
        )
    finally:
        db.close()

    ranked = demote(candidates, penalties, k=k, now=now or datetime.utcnow())
    return [documents[document_id] for document_id, _score in ranked]


class DemotingRetriever(BaseRetriever):
    """A retriever that applies feedback demotion, for the LangChain chain."""

    k: int = DEFAULT_K

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return retrieve_with_demotion(query, k=self.k)


def get_retriever():
    """The retriever the conversational chain uses."""
    return DemotingRetriever()

# Bounded LRU of per-session memory so a long-running bot does not leak RAM
# as new sessions appear. Oldest sessions are evicted past MAX_SESSION_MEMORIES.
MAX_SESSION_MEMORIES = int(os.getenv("MAX_SESSION_MEMORIES", "1000"))
session_memories: OrderedDict[str, ConversationBufferMemory] = OrderedDict()

def get_memory(session_id: str):
    if session_id in session_memories:
        session_memories.move_to_end(session_id)
    else:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        if len(session_memories) > MAX_SESSION_MEMORIES:
            session_memories.popitem(last=False)
    return session_memories[session_id]

def get_conversational_chain(session_id: str):
    llm = ChatOpenAI(temperature=0)
    retriever = get_retriever()
    memory = get_memory(session_id)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return chain

def generate_response(session_id: str, question: str):
    chain = get_conversational_chain(session_id)
    result = chain.invoke({"question": question})
    
    answer = result['answer']
    source_docs = result.get('source_documents', [])
    
    # Extract source document IDs
    doc_ids = [doc.metadata.get('id', 'unknown') for doc in source_docs]
    
    return answer, doc_ids
