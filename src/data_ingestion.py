# src/data_ingestion.py
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader
from typing import List, Optional
import os


def load_text_file(path: str, verbose: bool = True) -> List[Document]:
    """
    Load a single text file into a list of LangChain Document objects.

    Args:
        path: Path to the text file.
        verbose: If True, prints debug information.

    Returns:
        List of Document objects.
    """
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()

    if verbose:
        print(f"Loaded {len(documents)} document(s) from {path}")

    return documents


def load_directory(dir_path: str, file_types: Optional[List[str]] = None, verbose: bool = True) -> List[Document]:
    """
    Load all documents of specified types from a directory.

    Args:
        dir_path: Path to the directory to scan.
        file_types: List of file types to load (e.g., ['txt', 'pdf']).
                    Defaults to ['txt', 'pdf'].
        verbose: If True, prints debug information.

    Returns:
        List of Document objects with metadata including 'file_type' and 'filename'.
    """
    if file_types is None:
        file_types = ['txt', 'pdf']

    all_docs: List[Document] = []

    try:
        # Load text files
        if 'txt' in file_types:
            text_loader = DirectoryLoader(
                dir_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
            )
            text_docs = text_loader.load()
            for doc in text_docs:
                doc.metadata['file_type'] = 'txt'
                doc.metadata['filename'] = os.path.basename(doc.metadata.get('source', 'unknown'))
            all_docs.extend(text_docs)
            if verbose:
                print(f"{len(text_docs)} text documents loaded from {dir_path}")

        # Load PDF files
        if 'pdf' in file_types:
            pdf_loader = DirectoryLoader(
                dir_path,
                glob="**/*.pdf",
                loader_cls=lambda path: PyMuPDFLoader(path, mode="single"),  # combine pages
            )
            pdf_docs = pdf_loader.load()
            for doc in pdf_docs:
                doc.metadata['file_type'] = 'pdf'
                doc.metadata['filename'] = os.path.basename(doc.metadata.get('source', 'unknown'))
            all_docs.extend(pdf_docs)
            if verbose:
                print(f"{len(pdf_docs)} PDF documents loaded from {dir_path}")

        if verbose:
            print(f"Total documents loaded: {len(all_docs)}\n")

    except Exception as e:
        print(f"Error loading documents: {e}")

    return all_docs
