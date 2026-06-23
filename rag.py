import os
from collections import OrderedDict
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
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

def get_retriever():
    # Retrieve documents. We could adjust search_kwargs based on feedback.
    return get_vectorstore().as_retriever(search_kwargs={"k": 4})

# Bounded LRU of per-session memory so a long-running bot does not leak RAM
# as new sessions appear. Oldest sessions are evicted past MAX_SESSION_MEMORIES.
MAX_SESSION_MEMORIES = int(os.getenv("MAX_SESSION_MEMORIES", "1000"))
session_memories = OrderedDict()

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
