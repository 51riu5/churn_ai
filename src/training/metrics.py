"""
Evaluation metrics with optimal threshold selection.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn import metrics


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the probability threshold that maximizes F1 score."""
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )
    # thresholds has len = len(precisions) - 1
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute common binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = metrics.average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
    }


def classification_metrics_optimized(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute metrics using the F1-optimal threshold."""
    best_t = optimal_threshold(y_true, y_prob)
    return classification_metrics(y_true, y_prob, threshold=best_t)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return metrics.confusion_matrix(y_true, y_pred)
