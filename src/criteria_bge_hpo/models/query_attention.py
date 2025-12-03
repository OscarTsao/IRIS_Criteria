"""Learnable query vectors and attention mechanism for IRIS."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryVectors(nn.Module):
    """
    Learnable query vectors for retrieving task-relevant chunks.

    Each query specializes to retrieve different types of information
    (e.g., different symptom patterns in clinical text).
    """

    def __init__(
        self,
        num_queries: int = 8,
        embedding_dim: int = 768,
        init_strategy: str = "random",
    ):
        """
        Initialize query vectors.

        Args:
            num_queries: Number of learnable queries (default: 8)
            embedding_dim: Dimension of embeddings (default: 768)
            init_strategy: Initialization strategy ("random" or "orthogonal")
        """
        super().__init__()

        self.num_queries = num_queries
        self.embedding_dim = embedding_dim

        # Initialize query vectors
        self.queries = nn.Parameter(torch.randn(num_queries, embedding_dim))

        if init_strategy == "orthogonal":
            nn.init.orthogonal_(self.queries)
        elif init_strategy == "xavier":
            nn.init.xavier_normal_(self.queries)

    def forward(self) -> torch.Tensor:
        """
        Return query vectors.

        Returns:
            Query vectors [num_queries, embedding_dim]
        """
        return self.queries

    def get_normalized_queries(self) -> torch.Tensor:
        """
        Return L2-normalized query vectors.

        Returns:
            Normalized queries [num_queries, embedding_dim]
        """
        return F.normalize(self.queries, p=2, dim=1)


class LinearAttention(nn.Module):
    """
    Linear attention mechanism for aggregating retrieved chunks.

    Uses temperature-scaled softmax for sharp, interpretable attention.
    """

    def __init__(
        self,
        temperature: float = 0.1,
    ):
        """
        Initialize linear attention.

        Args:
            temperature: Softmax temperature (default: 0.1 for sharp attention)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        retrieved_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted aggregation.

        Args:
            query: Query vector [embedding_dim]
            retrieved_embeddings: Retrieved chunk embeddings [k, embedding_dim]

        Returns:
            Aggregated vector [embedding_dim]
        """
        # Compute attention weights: dot(query, chunks) / temperature
        # query: [embedding_dim]
        # retrieved_embeddings: [k, embedding_dim]
        attention_scores = torch.matmul(retrieved_embeddings, query) / self.temperature

        # Softmax normalization
        attention_weights = F.softmax(attention_scores, dim=0)  # [k]

        # Weighted sum
        aggregated = torch.matmul(attention_weights, retrieved_embeddings)  # [embedding_dim]

        return aggregated


class QueryPenaltyLoss(nn.Module):
    """
    Query penalty loss to encourage diversity among query vectors.

    Penalizes high cosine similarity between different queries.
    """

    def __init__(
        self,
        lambda_penalty: float = 0.1,
        similarity_threshold: float = 0.4,
    ):
        """
        Initialize query penalty loss.

        Args:
            lambda_penalty: Weight for penalty term (default: 0.1)
            similarity_threshold: Similarity above which to apply penalty (default: 0.4)
        """
        super().__init__()
        self.lambda_penalty = lambda_penalty
        self.similarity_threshold = similarity_threshold

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Compute query penalty loss.

        Args:
            queries: Query vectors [num_queries, embedding_dim]

        Returns:
            Penalty loss (scalar)
        """
        # Normalize queries
        queries_norm = F.normalize(queries, p=2, dim=1)  # [num_queries, embedding_dim]

        # Compute pairwise cosine similarities
        similarity_matrix = torch.matmul(queries_norm, queries_norm.T)  # [num_queries, num_queries]

        # Remove diagonal (self-similarity)
        num_queries = queries.size(0)
        mask = torch.eye(num_queries, device=queries.device)
        similarity_matrix = similarity_matrix * (1 - mask)

        # Penalize similarities above threshold
        penalty = F.relu(similarity_matrix - self.similarity_threshold).sum()

        return self.lambda_penalty * penalty


class QueryAttentionModule(nn.Module):
    """
    Complete query-attention module combining queries and linear attention.
    """

    def __init__(
        self,
        num_queries: int = 8,
        embedding_dim: int = 768,
        temperature: float = 0.1,
        init_strategy: str = "random",
    ):
        """
        Initialize query-attention module.

        Args:
            num_queries: Number of learnable queries
            embedding_dim: Dimension of embeddings
            temperature: Softmax temperature
            init_strategy: Query initialization strategy
        """
        super().__init__()

        self.query_vectors = QueryVectors(num_queries, embedding_dim, init_strategy)
        self.attention = LinearAttention(temperature)
        self.num_queries = num_queries

    def forward(
        self,
        retrieved_chunks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attention to retrieved chunks for all queries.

        Args:
            retrieved_chunks: Retrieved embeddings [num_queries, k, embedding_dim]

        Returns:
            Aggregated vectors [num_queries, embedding_dim]
        """
        queries = self.query_vectors()

        aggregated = []
        for i in range(self.num_queries):
            query = queries[i]  # [embedding_dim]
            chunks = retrieved_chunks[i]  # [k, embedding_dim]

            agg = self.attention(query, chunks)
            aggregated.append(agg)

        aggregated = torch.stack(aggregated, dim=0)  # [num_queries, embedding_dim]

        return aggregated
