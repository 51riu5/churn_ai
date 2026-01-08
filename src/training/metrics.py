from typing import Dict, Tuple

import numpy as np
from sklearn import metrics


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
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return metrics.confusion_matrix(y_true, y_pred)

