"""Evaluation metrics and interpretability analysis."""

from .evaluator import (
    BinaryClassificationMetrics,
    IRISInterpretabilityAnalyzer,
    PerCriterionEvaluator,
    compute_aggregate_metrics,
)

__all__ = [
    "BinaryClassificationMetrics",
    "PerCriterionEvaluator",
    "IRISInterpretabilityAnalyzer",
    "compute_aggregate_metrics",
]
