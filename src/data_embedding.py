# src/data_embedding.py
from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    Handles embedding of text using a SentenceTransformer model.

    Attributes:
        model_name: Name of the SentenceTransformer model.
        model: Loaded SentenceTransformer model instance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the EmbeddingManager and load the model.

        Args:
            model_name: Hugging Face / SentenceTransformers model name.
        """
        self.model_name: str = model_name
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            # print(f"{self.model_name} embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading embedding model '{self.model_name}': {e}")
            raise

    def generate_embeddings(self, texts: List[str], verbose: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            verbose: If True, display a progress bar during embedding.

        Returns:
            Numpy array of shape (len(texts), embedding_dim) containing embeddings.
        """
        if not texts:
            return np.array([])

        try:
            embeddings: np.ndarray
            if verbose:
                embeddings = self.model.encode(texts, show_progress_bar=True)
            else:
                embeddings = self.model.encode(texts)
            # print(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
