# src/rag_retriever.py
from __future__ import annotations
from typing import List

from src.data_embedding import EmbeddingManager
from src.data_vector_store import VectorStore
from src.types import RetrievedDocument


class RAGRetriever:
    """
    Retrieves top-k relevant documents from a vector store for a given query
    using embedding similarity.
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager) -> None:
        """
        Args:
            vector_store: Instance of VectorStore for document search.
            embedding_manager: Instance of EmbeddingManager to generate query embeddings.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[RetrievedDocument]:
        """
        Retrieve top-k relevant documents for a query with similarity scores.

        Args:
            query: Text query to search for.
            top_k: Maximum number of documents to retrieve.
            score_threshold: Minimum similarity score to include a document.

        Returns:
            List of RetrievedDocument objects, sorted by rank.
        """
        if not query.strip():
            return []

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs: List[RetrievedDocument] = []

            if not results['documents'] or not results['documents'][0]:
                return []

            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]

            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                similarity_score = 1.0 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append(
                        RetrievedDocument(
                            id=doc_id,
                            metadata=metadata,
                            document=document,
                            similarity_score=similarity_score,
                            distance=distance,
                            rank=i + 1
                        )
                    )

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
