"""
Publication-quality visualizations for churn prediction project.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

# ── Style ────────────────────────────────────────────────────────────
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2")
MODEL_COLORS = {
    "LogReg": PALETTE[0],
    "RF": PALETTE[1],
    "LSTM": PALETTE[2],
    "Transformer": PALETTE[3],
    "MSTAN (Ours)": PALETTE[4],
}


# ── Individual plots ────────────────────────────────────────────────


def plot_confusion_matrix(cm: np.ndarray, class_names, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax, annot_kws={"size": 16},
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, path: str) -> None:
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        auc = metrics.auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, color=PALETTE[4], label=f"ROC AUC = {auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE[4])
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ValueError:
        return


def plot_attention_weights(
    attn: np.ndarray, path: str, title: str = "Attention Heatmap"
) -> None:
    if attn.ndim == 4:
        attn = attn[0]
    mean_attn = attn.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mean_attn, cmap="magma", ax=ax, square=True)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Key Positions", fontsize=11)
    ax.set_ylabel("Query Positions", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Comparative dashboard plots ────────────────────────────────────


def plot_model_comparison_bar(
    model_metrics: Dict[str, Dict[str, float]], path: str
) -> None:
    """Grouped bar chart comparing all models across key metrics."""
    metric_keys = ["f1", "recall", "precision", "roc_auc", "pr_auc"]
    metric_labels = ["F1 Score", "Recall", "Precision", "ROC-AUC", "PR-AUC"]
    model_names = list(model_metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(model_names):
        vals = [model_metrics[model_name].get(k, 0) for k in metric_keys]
        color = MODEL_COLORS.get(model_name, PALETTE[i % len(PALETTE)])
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Test Set Performance", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multi_roc(
    model_results: Dict[str, Dict],  # {name: {"y_true": arr, "y_prob": arr}}
    path: str,
) -> None:
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, data) in enumerate(model_results.items()):
        y_true, y_prob = data["y_true"], data["y_prob"]
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        auc = metrics.auc(fpr, tpr)
        color = MODEL_COLORS.get(name, PALETTE[i % len(PALETTE)])
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multi_pr(
    model_results: Dict[str, Dict],
    path: str,
) -> None:
    """Overlay Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, data) in enumerate(model_results.items()):
        y_true, y_prob = data["y_true"], data["y_prob"]
        prec, rec, _ = metrics.precision_recall_curve(y_true, y_prob)
        ap = metrics.average_precision_score(y_true, y_prob)
        color = MODEL_COLORS.get(name, PALETTE[i % len(PALETTE)])
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    path: str,
) -> None:
    """2x2 grid: train loss, val loss, val F1, val AUC across models."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    panels = [
        ("train_loss", "Train Loss", axes[0, 0]),
        ("val_loss", "Validation Loss", axes[0, 1]),
        ("val_f1", "Validation F1", axes[1, 0]),
        ("val_auc", "Validation ROC-AUC", axes[1, 1]),
    ]

    for key, title, ax in panels:
        for i, (name, hist) in enumerate(histories.items()):
            if key in hist:
                color = MODEL_COLORS.get(name, PALETTE[i % len(PALETTE)])
                ax.plot(hist[key], lw=2, color=color, label=name)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Training Dynamics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_radar_chart(
    model_metrics: Dict[str, Dict[str, float]], path: str,
) -> None:
    """Spider/radar chart comparing models across metrics."""
    metric_keys = ["f1", "recall", "precision", "roc_auc", "pr_auc"]
    metric_labels = ["F1", "Recall", "Precision", "ROC-AUC", "PR-AUC"]
    n = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (name, mets) in enumerate(model_metrics.items()):
        vals = [mets.get(k, 0) for k in metric_keys]
        vals += vals[:1]
        color = MODEL_COLORS.get(name, PALETTE[i % len(PALETTE)])
        ax.plot(angles, vals, "o-", lw=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Performance Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
