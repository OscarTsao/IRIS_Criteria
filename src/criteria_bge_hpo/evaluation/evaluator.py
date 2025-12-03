"""Evaluation metrics for binary classification and interpretability analysis.

Implements:
- Binary classification metrics (F1, precision, recall, AUC)
- Per-criterion performance tracking
- Confusion matrices and ROC curves
- IRIS interpretability analysis
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import pandas as pd


class BinaryClassificationMetrics:
    """Compute comprehensive binary classification metrics.

    Metrics:
    - Accuracy, Precision, Recall (Sensitivity), Specificity
    - F1 Score (binary and macro)
    - AUC-ROC, AUC-PR (Average Precision)
    - Confusion Matrix
    """

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all binary classification metrics.

        Args:
            y_true: [num_samples] - Ground truth labels (0 or 1)
            y_pred: [num_samples] - Predicted labels (0 or 1)
            y_prob: [num_samples] - Predicted probabilities (optional, for AUC)

        Returns:
            metrics: Dict with all computed metrics
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["sensitivity"] = metrics["recall"]  # Alias
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        # Macro F1 (average of pos/neg F1)
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["true_positive"] = int(tp)
        metrics["true_negative"] = int(tn)
        metrics["false_positive"] = int(fp)
        metrics["false_negative"] = int(fn)

        # AUC metrics (require probabilities)
        if y_prob is not None:
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["auc_roc"] = 0.0

            try:
                metrics["auc_pr"] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics["auc_pr"] = 0.0

        return metrics

    @staticmethod
    def compute_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            normalize: If True, normalize by row sums

        Returns:
            cm: [2, 2] confusion matrix
                [[TN, FP],
                 [FN, TP]]
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        return cm

    @staticmethod
    def compute_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve.

        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities

        Returns:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Threshold values
        """
        return roc_curve(y_true, y_prob)

    @staticmethod
    def compute_precision_recall_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve.

        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities

        Returns:
            precision: Precision values
            recall: Recall values
            thresholds: Threshold values
        """
        return precision_recall_curve(y_true, y_prob)


class PerCriterionEvaluator:
    """Evaluate performance per DSM-5 criterion.

    Tracks metrics for each criterion separately to identify
    which criteria are easy/hard to classify.
    """

    def __init__(self, criterion_names: List[str]):
        """
        Initialize evaluator.

        Args:
            criterion_names: List of criterion identifiers (e.g., ['A.1', 'A.2', ...])
        """
        self.criterion_names = criterion_names
        self.reset()

    def reset(self):
        """Reset all stored predictions and labels."""
        self.predictions = {name: [] for name in self.criterion_names}
        self.probabilities = {name: [] for name in self.criterion_names}
        self.labels = {name: [] for name in self.criterion_names}

    def update(
        self,
        criterion_ids: List[str],
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ):
        """
        Update with new predictions.

        Args:
            criterion_ids: [batch_size] - Criterion IDs for each sample
            y_pred: [batch_size] - Predicted labels
            y_true: [batch_size] - Ground truth labels
            y_prob: [batch_size] - Predicted probabilities (optional)
        """
        for i, criterion_id in enumerate(criterion_ids):
            if criterion_id in self.predictions:
                self.predictions[criterion_id].append(y_pred[i])
                self.labels[criterion_id].append(y_true[i])
                if y_prob is not None:
                    self.probabilities[criterion_id].append(y_prob[i])

    def compute_metrics(self) -> pd.DataFrame:
        """
        Compute metrics for each criterion.

        Returns:
            df: DataFrame with columns [criterion, accuracy, precision, recall, f1, support]
        """
        results = []

        for criterion_name in self.criterion_names:
            if len(self.labels[criterion_name]) == 0:
                continue

            y_true = np.array(self.labels[criterion_name])
            y_pred = np.array(self.predictions[criterion_name])
            y_prob = (
                np.array(self.probabilities[criterion_name])
                if self.probabilities[criterion_name]
                else None
            )

            # Compute metrics
            metrics = BinaryClassificationMetrics.compute_all_metrics(
                y_true, y_pred, y_prob
            )

            results.append(
                {
                    "criterion": criterion_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "macro_f1": metrics["macro_f1"],
                    "auc_roc": metrics.get("auc_roc", 0.0),
                    "support": len(y_true),
                    "num_positive": int(y_true.sum()),
                }
            )

        return pd.DataFrame(results)

    def get_worst_criteria(self, metric: str = "f1", n: int = 3) -> List[str]:
        """
        Get N criteria with worst performance.

        Args:
            metric: Metric to rank by ('f1', 'accuracy', etc.)
            n: Number of worst criteria to return

        Returns:
            criterion_names: List of N worst criterion names
        """
        df = self.compute_metrics()
        if len(df) == 0:
            return []

        worst = df.nsmallest(n, metric)["criterion"].tolist()
        return worst

    def get_best_criteria(self, metric: str = "f1", n: int = 3) -> List[str]:
        """Get N criteria with best performance."""
        df = self.compute_metrics()
        if len(df) == 0:
            return []

        best = df.nlargest(n, metric)["criterion"].tolist()
        return best


class IRISInterpretabilityAnalyzer:
    """Analyze IRIS model interpretability.

    Extracts and analyzes:
    - Retrieved chunks per query
    - Query specialization (which queries focus on which symptoms)
    - Attention weights over retrieved chunks
    """

    def __init__(self, model):
        """
        Initialize analyzer.

        Args:
            model: IRIS model instance (must have retriever and query_attention)
        """
        self.model = model
        self.num_queries = model.query_attention.num_queries

    def get_retrieved_chunks_for_sample(
        self,
        post_text: str,
        criterion_text: str,
    ) -> Dict[int, List[str]]:
        """
        Get retrieved chunks for a single sample.

        Args:
            post_text: Post text
            criterion_text: Criterion text

        Returns:
            retrieved_chunks: {query_idx: [chunk1, chunk2, ...]}
        """
        if not hasattr(self.model, "retriever"):
            raise ValueError("Model has no retriever. Did you call build_retriever()?")

        # Get query vectors
        queries = self.model.query_attention.query_vectors.get_normalized_queries()

        # Search for each query
        retrieved_chunks = {}
        for query_idx in range(self.num_queries):
            query = queries[query_idx].unsqueeze(0)  # [1, dim]

            # Retrieve chunks
            similarities, indices = self.model.retriever.search(
                query.cpu().numpy(), k=self.model.k_retrieved
            )

            # Get chunk texts
            chunks = self.model.retriever.get_retrieved_chunks(indices)[0]
            retrieved_chunks[query_idx] = chunks

        return retrieved_chunks

    def analyze_query_specialization(
        self,
        sample_texts: List[Tuple[str, str]],
        criterion_names: Optional[List[str]] = None,
    ) -> Dict[int, Dict[str, int]]:
        """
        Analyze which queries specialize to which criteria.

        Args:
            sample_texts: List of (post_text, criterion_text) tuples
            criterion_names: Optional list of criterion names

        Returns:
            specialization: {query_idx: {criterion_name: count}}
        """
        # Count which queries retrieve chunks most relevant to each criterion
        specialization = {i: {} for i in range(self.num_queries)}

        for idx, (post_text, criterion_text) in enumerate(sample_texts):
            criterion_name = (
                criterion_names[idx] if criterion_names else f"Sample_{idx}"
            )

            # Get retrieved chunks
            retrieved = self.get_retrieved_chunks_for_sample(
                post_text, criterion_text
            )

            # For each query, check if chunks contain criterion keywords
            for query_idx, chunks in retrieved.items():
                # Simple keyword matching (can be improved)
                criterion_keywords = criterion_text.lower().split()[:3]
                relevant = sum(
                    1
                    for chunk in chunks
                    if any(kw in chunk.lower() for kw in criterion_keywords)
                )

                if criterion_name not in specialization[query_idx]:
                    specialization[query_idx][criterion_name] = 0
                specialization[query_idx][criterion_name] += relevant

        return specialization

    def get_attention_weights(
        self,
        post_text: str,
        criterion_text: str,
    ) -> Dict[int, np.ndarray]:
        """
        Get attention weights for retrieved chunks.

        Args:
            post_text: Post text
            criterion_text: Criterion text

        Returns:
            attention_weights: {query_idx: [k] weights}
        """
        if not hasattr(self.model, "retriever"):
            raise ValueError("Model has no retriever.")

        # Get query vectors
        queries = self.model.query_attention.query_vectors.get_normalized_queries()

        # Get chunk embeddings for this post
        # NOTE: This requires the post to be in the retriever's index
        attention_weights = {}

        for query_idx in range(self.num_queries):
            query = queries[query_idx]

            # Retrieve chunks
            similarities, indices = self.model.retriever.search(
                query.unsqueeze(0).cpu().numpy(), k=self.model.k_retrieved
            )

            # Get chunk embeddings
            chunk_embeddings = torch.from_numpy(
                self.model.retriever.index.reconstruct_n(0, self.model.retriever.num_chunks)
            )[indices[0]]

            # Compute attention weights
            with torch.no_grad():
                weights = self.model.query_attention.attention(
                    query, chunk_embeddings
                )

            attention_weights[query_idx] = weights.cpu().numpy()

        return attention_weights

    def visualize_query_focus(
        self,
        post_text: str,
        criterion_text: str,
        top_k: int = 3,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Visualize which chunks each query focuses on.

        Args:
            post_text: Post text
            criterion_text: Criterion text
            top_k: Number of top chunks to return per query

        Returns:
            focus: {query_idx: [(chunk_text, attention_weight), ...]}
        """
        retrieved_chunks = self.get_retrieved_chunks_for_sample(
            post_text, criterion_text
        )
        attention_weights = self.get_attention_weights(post_text, criterion_text)

        focus = {}
        for query_idx in range(self.num_queries):
            chunks = retrieved_chunks[query_idx]
            weights = attention_weights[query_idx]

            # Sort by attention weight
            sorted_indices = np.argsort(weights)[::-1][:top_k]
            focus[query_idx] = [
                (chunks[i], weights[i]) for i in sorted_indices
            ]

        return focus


def compute_aggregate_metrics(
    all_predictions: List[np.ndarray],
    all_labels: List[np.ndarray],
    all_probabilities: Optional[List[np.ndarray]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute aggregate metrics across multiple folds.

    Args:
        all_predictions: List of prediction arrays (one per fold)
        all_labels: List of label arrays (one per fold)
        all_probabilities: List of probability arrays (optional)

    Returns:
        metrics: {metric_name: (mean, std)}
    """
    fold_metrics = []

    for fold_idx in range(len(all_predictions)):
        y_pred = all_predictions[fold_idx]
        y_true = all_labels[fold_idx]
        y_prob = all_probabilities[fold_idx] if all_probabilities else None

        metrics = BinaryClassificationMetrics.compute_all_metrics(
            y_true, y_pred, y_prob
        )
        fold_metrics.append(metrics)

    # Compute mean and std for each metric
    aggregate = {}
    for metric_name in fold_metrics[0].keys():
        values = [fold[metric_name] for fold in fold_metrics]
        aggregate[metric_name] = (np.mean(values), np.std(values))

    return aggregate
