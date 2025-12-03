"""FAISS-based retrieval mechanism for IRIS."""

from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ChunkRetriever:
    """
    FAISS-based retrieval system for efficient similarity search.

    Stores pre-computed chunk embeddings and retrieves k most similar
    chunks for each query vector.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        use_gpu: bool = False,
    ):
        """
        Initialize retriever.

        Args:
            embedding_dim: Dimension of embeddings (e.g., 768 for many encoders)
            use_gpu: Whether to use GPU for FAISS index
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu

        # Create FAISS index (inner product = cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)

        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.chunk_texts = []
        self.chunk_metadata = []

    def add_chunks(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Add chunks to the index.

        Args:
            embeddings: Chunk embeddings [num_chunks, embedding_dim]
            texts: Original chunk texts
            metadata: Optional metadata for each chunk
        """
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Store texts and metadata
        self.chunk_texts.extend(texts)
        if metadata is not None:
            self.chunk_metadata.extend(metadata)
        else:
            self.chunk_metadata.extend([{}] * len(texts))

    def search(
        self,
        query_vectors: torch.Tensor,
        k: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k chunks for each query.

        Args:
            query_vectors: Query vectors [num_queries, embedding_dim]
            k: Number of chunks to retrieve per query

        Returns:
            Tuple of (similarities, indices)
            - similarities: [num_queries, k] cosine similarities
            - indices: [num_queries, k] chunk indices
        """
        # Convert to numpy and normalize
        query_np = query_vectors.detach().cpu().numpy()
        query_np = query_np / np.linalg.norm(query_np, axis=1, keepdims=True)

        # Search
        similarities, indices = self.index.search(query_np.astype(np.float32), k)

        # Convert back to tensors
        similarities = torch.from_numpy(similarities).to(query_vectors.device)
        indices = torch.from_numpy(indices).to(query_vectors.device)

        return similarities, indices

    def get_retrieved_chunks(
        self,
        indices: torch.Tensor,
    ) -> List[List[str]]:
        """
        Get chunk texts for retrieved indices.

        Args:
            indices: Retrieved indices [num_queries, k]

        Returns:
            List of lists of chunk texts
        """
        retrieved = []
        for query_indices in indices:
            query_chunks = [self.chunk_texts[idx] for idx in query_indices.cpu().numpy()]
            retrieved.append(query_chunks)
        return retrieved

    def reset(self):
        """Clear the index and stored chunks."""
        self.index.reset()
        self.chunk_texts = []
        self.chunk_metadata = []

    @property
    def num_chunks(self) -> int:
        """Return number of chunks in index."""
        return self.index.ntotal


class EmbeddingModel(nn.Module):
    """
    Wrapper for sentence embedding models (e.g., Sentence-BERT).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        freeze: bool = True,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name or Sentence-Transformers model
            freeze: Whether to freeze encoder weights (default: True for IRIS)
        """
        super().__init__()

        # Load sentence transformer
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Embeddings [num_texts, embedding_dim]
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
        )
        return embeddings

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Forward pass for training mode."""
        return self.encode(texts)
