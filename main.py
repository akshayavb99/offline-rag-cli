from __future__ import annotations

from pathlib import Path
import os
import argparse

from dotenv import load_dotenv

from src.data_ingestion import load_directory
from src.data_chunking import split_documents
from src.data_embedding import EmbeddingManager
from src.data_vector_store import VectorStore
from src.rag_retriever import RAGRetriever
from src.integrate_llm import OllamaChat
from langchain_core.documents import Document


def main() -> None:
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="RAG Assistant CLI")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex documents and rebuild vector store (slow, optional)"
    )
    args = parser.parse_args()

    # -----------------------------
    # Load .env variables and Setup Data Directory Paths
    # -----------------------------
    load_dotenv()  # loads .env from project root
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
    if not OLLAMA_MODEL_NAME:
        raise ValueError("OLLAMA_MODEL_NAME not found in .env")
    print("Using Ollama model:", OLLAMA_MODEL_NAME)
    OLLAMA_CONTAINER_NAME = os.getenv("OLLAMA_CONTAINER_NAME")
    OLLAMA_PORT = os.getenv("OLLAMA_PORT")

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
    VECTOR_STORE_DIR = DATA_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store")
    OLLAMA_DATA_DIR = DATA_DIR / os.getenv("OLLAMA_DATA_DIR", "ollama_data")
    

    # -----------------------------
    # Load documents and build vector store if requested
    # -----------------------------
    if args.reindex or not VECTOR_STORE_DIR.exists():
        print("Reindexing documents...")
        docs_list: list[Document] = load_directory(DATA_DIR)
        doc_chunks: list[Document] = split_documents(docs_list)

        embedding_manager = EmbeddingManager()
        doc_chunk_texts = [doc.page_content for doc in doc_chunks]
        embeddings = embedding_manager.generate_embeddings(doc_chunk_texts, verbose=True)

        vector_store = VectorStore(persist_directory=VECTOR_STORE_DIR)
        vector_store.add_documents(doc_chunks, embeddings)
    else:
        print("Loading existing vector store...")
        vector_store = VectorStore(persist_directory=VECTOR_STORE_DIR)
        embedding_manager = EmbeddingManager()  # needed for retriever

    # -----------------------------
    # Initialize RAG retriever
    # -----------------------------
    rag_retriever = RAGRetriever(vector_store, embedding_manager)

    # -----------------------------
    # Initialize Ollama Chat
    # -----------------------------
    ollama_chat = OllamaChat(model=OLLAMA_MODEL_NAME, retriever=rag_retriever, container_name = OLLAMA_CONTAINER_NAME, port = OLLAMA_PORT, data_volume = OLLAMA_DATA_DIR)
    print("Assistant:", ollama_chat.welcome_message)

    # -----------------------------
    # CLI Loop
    # -----------------------------
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "end"}:
            print("Goodbye!")
            ollama_chat._stop_container()
            break

        print("Assistant: ", end="", flush=True)
        try:
            for chunk in ollama_chat.chat(query, stream=True):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n[Error] {e}")
        print()  # newline after assistant response


if __name__ == "__main__":
    main()
