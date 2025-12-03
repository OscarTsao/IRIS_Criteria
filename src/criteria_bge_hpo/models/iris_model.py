"""IRIS model for interpretable retrieval-augmented classification."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_attention import QueryAttentionModule, QueryPenaltyLoss
from .retrieval import ChunkRetriever, EmbeddingModel


class IRISClassifier(nn.Module):
    """
    IRIS: Interpretable Retrieval-Augmented Classification.

    Architecture:
    1. Encode post chunks with frozen encoder
    2. Learnable query vectors retrieve k relevant chunks each
    3. Linear attention aggregates retrieved chunks per query
    4. MLP classifier predicts from aggregated representations
    """

    def __init__(
        self,
        num_queries: int = 8,
        k_retrieved: int = 8,
        embedding_dim: int = 768,
        temperature: float = 0.1,
        encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        query_penalty_lambda: float = 0.1,
        query_penalty_threshold: float = 0.4,
    ):
        """
        Initialize IRIS classifier.

        Args:
            num_queries: Number of learnable query vectors (default: 8)
            k_retrieved: Number of chunks to retrieve per query (default: 8)
            embedding_dim: Dimension of embeddings (default: 768)
            temperature: Softmax temperature for attention (default: 0.1)
            encoder_name: Pre-trained encoder model name
            num_classes: Number of output classes (default: 2 for binary)
            hidden_dim: Hidden dimension for MLP (default: 256)
            dropout: Dropout rate (default: 0.1)
            query_penalty_lambda: Query diversity penalty weight (default: 0.1)
            query_penalty_threshold: Similarity threshold for penalty (default: 0.4)
        """
        super().__init__()

        self.num_queries = num_queries
        self.k_retrieved = k_retrieved
        self.embedding_dim = embedding_dim

        # Embedding model (frozen)
        self.embedding_model = EmbeddingModel(encoder_name, freeze=True)

        # Query-attention module
        self.query_attention = QueryAttentionModule(
            num_queries=num_queries,
            embedding_dim=embedding_dim,
            temperature=temperature,
        )

        # Classification head
        input_dim = num_queries * embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Query penalty loss
        self.query_penalty = QueryPenaltyLoss(query_penalty_lambda, query_penalty_threshold)

        # Retriever (will be populated with chunks)
        self.retriever = None

    def build_retriever(
        self,
        chunk_texts: List[str],
        batch_size: int = 32,
        use_gpu: bool = False,
    ):
        """
        Build FAISS retriever with pre-computed chunk embeddings.

        Args:
            chunk_texts: List of chunk texts to index
            batch_size: Batch size for encoding
            use_gpu: Whether to use GPU for FAISS
        """
        # Encode chunks
        print(f"Encoding {len(chunk_texts)} chunks...")
        embeddings = self.embedding_model.encode(chunk_texts, batch_size=batch_size)

        # Create retriever
        self.retriever = ChunkRetriever(
            embedding_dim=self.embedding_dim,
            use_gpu=use_gpu,
        )

        # Add chunks to index
        self.retriever.add_chunks(
            embeddings.cpu().numpy(),
            chunk_texts,
        )
        print(f"Built retriever with {self.retriever.num_chunks} chunks")

    def forward(
        self,
        post_texts: Optional[List[str]] = None,
        criterion_texts: Optional[List[str]] = None,
        precomputed_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            post_texts: List of post texts (for encoding)
            criterion_texts: List of criterion texts (for encoding)
            precomputed_embeddings: Pre-computed embeddings [batch_size, embedding_dim]

        Returns:
            Dictionary with:
            - logits: [batch_size, num_classes]
            - query_penalty: Query diversity penalty loss
            - attention_weights: Retrieved chunk indices for interpretability
        """
        if self.retriever is None:
            raise ValueError("Retriever not built. Call build_retriever() first.")

        batch_size = len(post_texts) if post_texts is not None else precomputed_embeddings.size(0)

        # Get query vectors
        queries = self.query_attention.query_vectors.get_normalized_queries()
        # queries: [num_queries, embedding_dim]

        # Retrieve chunks for each query
        similarities, indices = self.retriever.search(queries, k=self.k_retrieved)
        # similarities: [num_queries, k]
        # indices: [num_queries, k]

        # Get embeddings of retrieved chunks
        # For simplicity, re-retrieve from index
        # In practice, could cache embeddings
        retrieved_chunks = []
        for query_idx in range(self.num_queries):
            chunk_indices = indices[query_idx].cpu().numpy()
            chunk_texts_for_query = [self.retriever.chunk_texts[idx] for idx in chunk_indices]

            # Encode retrieved chunks
            chunk_embeddings = self.embedding_model.encode(chunk_texts_for_query)
            retrieved_chunks.append(chunk_embeddings)

        retrieved_chunks = torch.stack(retrieved_chunks, dim=0)
        # retrieved_chunks: [num_queries, k, embedding_dim]

        # Apply attention to aggregate
        aggregated = self.query_attention(retrieved_chunks)
        # aggregated: [num_queries, embedding_dim]

        # Repeat for batch (all samples use same queries/retrieval for now)
        # TODO: Could make this sample-specific in future
        aggregated_batch = aggregated.unsqueeze(0).repeat(batch_size, 1, 1)
        # aggregated_batch: [batch_size, num_queries, embedding_dim]

        # Flatten and classify
        aggregated_flat = aggregated_batch.view(batch_size, -1)
        # aggregated_flat: [batch_size, num_queries * embedding_dim]

        logits = self.classifier(aggregated_flat)
        # logits: [batch_size, num_classes]

        # Compute query penalty
        query_penalty = self.query_penalty(self.query_attention.query_vectors())

        return {
            "logits": logits,
            "query_penalty": query_penalty,
            "retrieved_indices": indices,  # For interpretability
            "attention_scores": similarities,  # For interpretability
        }

    def predict(
        self,
        post_texts: List[str],
        criterion_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Make predictions.

        Args:
            post_texts: List of post texts
            criterion_texts: List of criterion texts (optional)

        Returns:
            Predictions [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(post_texts, criterion_texts)
            if outputs["logits"].size(1) == 2:
                # Binary classification
                probs = F.softmax(outputs["logits"], dim=1)
            else:
                # Multi-class
                probs = F.softmax(outputs["logits"], dim=1)
        return probs

    def get_retrieved_chunks_for_queries(
        self,
        post_text: str,
    ) -> Dict[int, List[str]]:
        """
        Get retrieved chunks for each query (for interpretability).

        Args:
            post_text: Post text

        Returns:
            Dictionary mapping query_idx to list of retrieved chunk texts
        """
        if self.retriever is None:
            raise ValueError("Retriever not built")

        queries = self.query_attention.query_vectors.get_normalized_queries()
        _, indices = self.retriever.search(queries, k=self.k_retrieved)

        retrieved = {}
        for query_idx in range(self.num_queries):
            chunk_indices = indices[query_idx].cpu().numpy()
            chunk_texts = [self.retriever.chunk_texts[idx] for idx in chunk_indices]
            retrieved[query_idx] = chunk_texts

        return retrieved


class IRISForCriterionMatching(nn.Module):
    """
    IRIS model adapted for post-criterion matching.

    Incorporates both post chunks and criterion text.
    """

    def __init__(
        self,
        num_queries: int = 8,
        k_retrieved: int = 8,
        embedding_dim: int = 768,
        temperature: float = 0.1,
        encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        query_penalty_lambda: float = 0.1,
        query_penalty_threshold: float = 0.4,
    ):
        """Initialize IRIS for criterion matching."""
        super().__init__()

        self.num_queries = num_queries
        self.k_retrieved = k_retrieved
        self.embedding_dim = embedding_dim

        # Embedding model
        self.embedding_model = EmbeddingModel(encoder_name, freeze=True)

        # Query-attention module
        self.query_attention = QueryAttentionModule(
            num_queries=num_queries,
            embedding_dim=embedding_dim,
            temperature=temperature,
        )

        # Query penalty loss
        self.query_penalty_loss = QueryPenaltyLoss(query_penalty_lambda, query_penalty_threshold)

        # Classification head (combines post aggregation + criterion)
        input_dim = num_queries * embedding_dim + embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

        self.query_penalty = QueryPenaltyLoss()
        self.retriever = None

    def build_retriever(
        self,
        chunk_texts: List[str],
        batch_size: int = 32,
        use_gpu: bool = False,
    ):
        """Build retriever for post chunks."""
        embeddings = self.embedding_model.encode(chunk_texts, batch_size=batch_size)

        self.retriever = ChunkRetriever(
            embedding_dim=self.embedding_dim,
            use_gpu=use_gpu,
        )
        self.retriever.add_chunks(embeddings.cpu().numpy(), chunk_texts)

    def forward(
        self,
        post_texts: List[str],
        criterion_texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for criterion matching.

        Args:
            post_texts: List of post texts
            criterion_texts: List of criterion texts

        Returns:
            Dictionary with logits and losses
        """
        if self.retriever is None:
            raise ValueError("Retriever not built")

        batch_size = len(post_texts)

        # Encode criteria
        criterion_embeddings = self.embedding_model.encode(criterion_texts)
        # criterion_embeddings: [batch_size, embedding_dim]

        # Get query vectors
        queries = self.query_attention.query_vectors.get_normalized_queries()

        # Retrieve chunks
        similarities, indices = self.retriever.search(queries, k=self.k_retrieved)

        # Get retrieved chunk embeddings
        retrieved_chunks = []
        for query_idx in range(self.num_queries):
            chunk_indices = indices[query_idx].cpu().numpy()
            chunk_texts_for_query = [self.retriever.chunk_texts[idx] for idx in chunk_indices]
            chunk_embeddings = self.embedding_model.encode(chunk_texts_for_query)
            retrieved_chunks.append(chunk_embeddings)

        retrieved_chunks = torch.stack(retrieved_chunks, dim=0)

        # Aggregate with attention
        aggregated = self.query_attention(retrieved_chunks)
        # aggregated: [num_queries, embedding_dim]

        # Repeat for batch
        aggregated_batch = aggregated.unsqueeze(0).repeat(batch_size, 1, 1)
        aggregated_flat = aggregated_batch.view(batch_size, -1)

        # Combine with criterion embeddings
        combined = torch.cat([aggregated_flat, criterion_embeddings], dim=1)

        # Classify
        logits = self.classifier(combined).squeeze(1)

        query_penalty = self.query_penalty(self.query_attention.query_vectors())

        return {
            "logits": logits,
            "query_penalty": query_penalty,
            "retrieved_indices": indices,
        }
