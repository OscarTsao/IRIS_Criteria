"""Tests for evaluation metrics and interpretability."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_binary_classification_metrics():
    """Test basic binary classification metrics."""
    from criteria_bge_hpo.evaluation import BinaryClassificationMetrics

    print("\n1. Testing BinaryClassificationMetrics...")

    # Perfect predictions
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.95])

    metrics = BinaryClassificationMetrics.compute_all_metrics(y_true, y_pred, y_prob)

    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"   F1: {metrics['f1']:.4f}")
    print(f"   Macro F1: {metrics['macro_f1']:.4f}")
    print(f"   Specificity: {metrics['specificity']:.4f}")
    print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"   AUC-PR: {metrics['auc_pr']:.4f}")

    assert metrics["accuracy"] == 1.0, "Perfect predictions should have accuracy=1.0"
    assert metrics["f1"] == 1.0, "Perfect predictions should have F1=1.0"

    # Imperfect predictions
    y_pred_imperfect = np.array([0, 1, 1, 0, 0, 1])  # 2 errors
    metrics_imperfect = BinaryClassificationMetrics.compute_all_metrics(
        y_true, y_pred_imperfect, y_prob
    )

    print(f"\n   Imperfect predictions:")
    print(f"   Accuracy: {metrics_imperfect['accuracy']:.4f}")
    print(f"   F1: {metrics_imperfect['f1']:.4f}")
    print(f"   TP={metrics_imperfect['true_positive']}, "
          f"TN={metrics_imperfect['true_negative']}, "
          f"FP={metrics_imperfect['false_positive']}, "
          f"FN={metrics_imperfect['false_negative']}")

    assert metrics_imperfect["accuracy"] < 1.0
    assert metrics_imperfect["f1"] < 1.0

    print("   ✓ BinaryClassificationMetrics passed")


def test_confusion_matrix():
    """Test confusion matrix computation."""
    from criteria_bge_hpo.evaluation import BinaryClassificationMetrics

    print("\n2. Testing confusion matrix...")

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])

    cm = BinaryClassificationMetrics.compute_confusion_matrix(y_true, y_pred)
    print(f"   Confusion matrix:\n{cm}")

    assert cm.shape == (2, 2), "Confusion matrix should be 2x2"
    assert cm[0, 0] == 2, "TN should be 2"
    assert cm[0, 1] == 1, "FP should be 1"
    assert cm[1, 0] == 1, "FN should be 1"
    assert cm[1, 1] == 2, "TP should be 2"

    # Normalized
    cm_norm = BinaryClassificationMetrics.compute_confusion_matrix(
        y_true, y_pred, normalize=True
    )
    print(f"   Normalized:\n{cm_norm}")

    assert np.allclose(cm_norm.sum(axis=1), [1.0, 1.0]), "Rows should sum to 1"

    print("   ✓ Confusion matrix passed")


def test_per_criterion_evaluator():
    """Test per-criterion evaluation."""
    from criteria_bge_hpo.evaluation import PerCriterionEvaluator

    print("\n3. Testing PerCriterionEvaluator...")

    # Create evaluator
    criteria = ["A.1", "A.2", "A.3"]
    evaluator = PerCriterionEvaluator(criteria)

    # Add predictions for different criteria
    # A.1: Perfect
    evaluator.update(
        criterion_ids=["A.1", "A.1", "A.1", "A.1"],
        y_pred=np.array([0, 0, 1, 1]),
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.2, 0.9, 0.8]),
    )

    # A.2: Some errors
    evaluator.update(
        criterion_ids=["A.2", "A.2", "A.2", "A.2"],
        y_pred=np.array([0, 1, 1, 0]),
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.3, 0.6, 0.9, 0.4]),
    )

    # A.3: All wrong
    evaluator.update(
        criterion_ids=["A.3", "A.3"],
        y_pred=np.array([1, 1]),
        y_true=np.array([0, 0]),
        y_prob=np.array([0.8, 0.9]),
    )

    # Compute metrics
    df = evaluator.compute_metrics()
    print(f"\n   Per-criterion metrics:")
    print(df.to_string(index=False))

    assert len(df) == 3, "Should have 3 criteria"
    assert df.loc[df["criterion"] == "A.1", "f1"].values[0] == 1.0, "A.1 should be perfect"
    assert df.loc[df["criterion"] == "A.3", "f1"].values[0] == 0.0, "A.3 should be all wrong"

    # Get worst criteria
    worst = evaluator.get_worst_criteria(metric="f1", n=1)
    print(f"\n   Worst criterion: {worst}")
    assert worst[0] == "A.3", "A.3 should be worst"

    # Get best criteria
    best = evaluator.get_best_criteria(metric="f1", n=1)
    print(f"   Best criterion: {best}")
    assert best[0] == "A.1", "A.1 should be best"

    print("   ✓ PerCriterionEvaluator passed")


def test_aggregate_metrics():
    """Test aggregate metrics across folds."""
    from criteria_bge_hpo.evaluation import compute_aggregate_metrics

    print("\n4. Testing aggregate metrics...")

    # Create dummy fold results
    all_predictions = [
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 0]),
        np.array([0, 0, 1, 1]),
    ]

    all_labels = [
        np.array([0, 0, 1, 1]),  # Fold 0: Perfect
        np.array([0, 0, 1, 1]),  # Fold 1: 50% accuracy
        np.array([0, 0, 1, 1]),  # Fold 2: Perfect
    ]

    all_probabilities = [
        np.array([0.1, 0.2, 0.9, 0.8]),
        np.array([0.3, 0.6, 0.9, 0.4]),
        np.array([0.1, 0.2, 0.9, 0.8]),
    ]

    # Compute aggregate
    aggregate = compute_aggregate_metrics(
        all_predictions, all_labels, all_probabilities
    )

    print(f"\n   Aggregate metrics (mean ± std):")
    for metric, (mean, std) in aggregate.items():
        if isinstance(mean, float):
            print(f"   {metric:15s}: {mean:.4f} ± {std:.4f}")

    assert "accuracy" in aggregate
    assert "f1" in aggregate
    assert "macro_f1" in aggregate

    # Check fold 0 and 2 have perfect accuracy, fold 1 has 0.5
    expected_acc = (1.0 + 0.5 + 1.0) / 3
    assert abs(aggregate["accuracy"][0] - expected_acc) < 0.01

    print("   ✓ Aggregate metrics passed")


def test_roc_curve():
    """Test ROC curve computation."""
    from criteria_bge_hpo.evaluation import BinaryClassificationMetrics

    print("\n5. Testing ROC curve...")

    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3])

    fpr, tpr, thresholds = BinaryClassificationMetrics.compute_roc_curve(
        y_true, y_prob
    )

    print(f"   FPR shape: {fpr.shape}")
    print(f"   TPR shape: {tpr.shape}")
    print(f"   Thresholds shape: {thresholds.shape}")
    print(f"   FPR range: [{fpr.min():.2f}, {fpr.max():.2f}]")
    print(f"   TPR range: [{tpr.min():.2f}, {tpr.max():.2f}]")

    assert len(fpr) == len(tpr) == len(thresholds)
    assert fpr[0] == 0.0 or fpr[-1] == 1.0, "FPR should start at 0 or end at 1"
    assert tpr[0] == 0.0 or tpr[-1] == 1.0, "TPR should start at 0 or end at 1"

    print("   ✓ ROC curve passed")


def test_precision_recall_curve():
    """Test precision-recall curve."""
    from criteria_bge_hpo.evaluation import BinaryClassificationMetrics

    print("\n6. Testing precision-recall curve...")

    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3])

    precision, recall, thresholds = BinaryClassificationMetrics.compute_precision_recall_curve(
        y_true, y_prob
    )

    print(f"   Precision shape: {precision.shape}")
    print(f"   Recall shape: {recall.shape}")
    print(f"   Thresholds shape: {thresholds.shape}")

    assert len(precision) == len(recall)
    assert len(thresholds) == len(precision) - 1

    print("   ✓ Precision-recall curve passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Evaluation Component Tests")
    print("=" * 70)

    try:
        test_binary_classification_metrics()
        test_confusion_matrix()
        test_per_criterion_evaluator()
        test_aggregate_metrics()
        test_roc_curve()
        test_precision_recall_curve()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
