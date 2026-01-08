from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

# Matplotlib backend suitable for headless environments
plt.switch_backend("Agg")


def plot_confusion_matrix(cm: np.ndarray, class_names, path: str) -> None:
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, path: str) -> None:
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except ValueError:
        # Happens when only one class present; skip plotting to avoid confusion.
        return


def plot_attention_weights(attn: np.ndarray, path: str, title: str = "Attention Heatmap") -> None:
    # attn assumed shape (heads, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
    if attn.ndim == 4:
        attn = attn[0]  # take first example
    mean_attn = attn.mean(axis=0)
    plt.figure(figsize=(5, 4))
    sns.heatmap(mean_attn, cmap="viridis")
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

