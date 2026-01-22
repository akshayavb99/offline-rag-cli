from typing import TypedDict, Any    

class RetrievedDocument(TypedDict):
    """
    Represents a document retrieved from the vector store for RAG.
    """
    id: str
    metadata: dict[str, Any]
    document: str
    similarity_score: float
    distance: float
    rank: int

class ChatMessage(TypedDict):
    """
    Typed representation of an Ollama chat message.
    """
    role: str
    content: str