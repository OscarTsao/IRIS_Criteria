"""Model implementations."""

from .classifier_heads import ClassifierHeadFactory
from .iris_model import IRISClassifier, IRISForCriterionMatching
from .query_attention import (
    LinearAttention,
    QueryAttentionModule,
    QueryPenaltyLoss,
    QueryVectors,
)
from .retrieval import ChunkRetriever, EmbeddingModel

__all__ = [
    # IRIS models
    "IRISClassifier",
    "IRISForCriterionMatching",
    # Classification heads
    "ClassifierHeadFactory",
    # Query attention
    "QueryVectors",
    "LinearAttention",
    "QueryAttentionModule",
    "QueryPenaltyLoss",
    # Retrieval
    "ChunkRetriever",
    "EmbeddingModel",
]
