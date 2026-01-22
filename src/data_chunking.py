# src/data_chunking.py
from __future__ import annotations
from typing import List
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True
) -> List[Document]:
    """
    Split documents into smaller chunks using recursive character splitting.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        verbose: If True, prints information about splitting.

    Returns:
        List of Document objects representing chunks.

    Notes:
        - Uses separators ["\\n\\n", "\\n", " ", ""] in order of priority.
        - Ensures chunks are smaller than chunk_size while preserving context.
        - Preserves document metadata for each chunk.
    """
    if not documents:
        if verbose:
            print("No documents provided to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    doc_chunks: List[Document] = text_splitter.split_documents(documents)

    if verbose:
        print(f"Split {len(documents)} documents into {len(doc_chunks)} chunks.")
        if doc_chunks:
            example_chunk: Document = doc_chunks[0]
            print(f"Example chunk (first 150 chars): {example_chunk.page_content[:150]!r}")
            print(f"Metadata: {example_chunk.metadata}")
        print()

    return doc_chunks
