from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from typing import Protocol, TypedDict

import chromadb
import numpy as np

class VectorStore:
    """
    Persistent vector store wrapper for RAG document embeddings.

    Uses ChromaDB as the backend and ensures deterministic document IDs
    to avoid duplicate storage across runs.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "../data/vector_store",
    ) -> None:
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Filesystem path for persistent storage.
        """
        self.collection_name: str = collection_name
        self.persist_directory: str = persist_directory

        self.client: chromadb.ClientAPI
        self.collection: chromadb.Collection

        self._initialize_store()

    def _initialize_store(self) -> None:
        """
        Create or load the persistent ChromaDB collection.

        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG"},
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(
                f"Number of documents in collection: "
                f"{self.collection.count()}"
            )
            print()

        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize vector store"
            ) from exc

    def add_documents(
        self,
        documents: Iterable[Document],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add documents and their embeddings to the vector store.

        Documents are deduplicated using a deterministic hash based on
        content and metadata.

        Args:
            documents: Iterable of document-like objects.
            embeddings: NumPy array of shape (n_documents, embedding_dim).

        Raises:
            ValueError: If document and embedding counts do not match.
        """
        documents_list = list(documents)

        if len(documents_list) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        print(
            f"Attempting to add {len(documents_list)} documents "
            "to the vector store"
        )

        ids: list[str] = []
        metadatas: list[StoredMetadata] = []
        documents_text: list[str] = []
        embeddings_list: list[list[float]] = []

        existing_ids = self._get_existing_ids()

        for index, (doc, embedding) in enumerate(
            zip(documents_list, embeddings)
        ):
            doc_id = self._generate_document_id(doc)

            if doc_id in existing_ids:
                continue

            ids.append(doc_id)
            metadatas.append(
                {
                    "doc_index": index,
                    "doc_length": len(doc.page_content),
                }
            )
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

            existing_ids.add(doc_id)

        if not ids:
            print("No new documents to add. All documents already exist.")
            print()
            return

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )

            print(f"Successfully added {len(ids)} new documents")
            print(
                f"Total documents in vector store: "
                f"{self.collection.count()}"
            )

        except Exception as exc:
            raise RuntimeError(
                "Failed to add documents to vector store"
            ) from exc

    def _get_existing_ids(self) -> set[str]:
        """
        Retrieve existing document IDs from the collection.
        """
        try:
            result = self.collection.get()
            return set(result["ids"])
        except Exception:
            return set()

    @staticmethod
    def _generate_document_id(doc: Document) -> str:
        """
        Generate a deterministic document ID.

        Args:
            doc: Document object.

        Returns:
            Deterministic document identifier string.
        """
        combined = f"{doc.page_content}{doc.metadata}"
        digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return f"doc_{digest[:16]}"
