"""
Focal Loss for class-imbalanced binary classification.

Standard BCE treats every sample equally, which causes the model to be
overconfident on the majority class (non-churners) and under-predict the
minority class (churners).  Focal Loss down-weights easy/well-classified
examples and focuses learning on hard negatives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

- gamma=0 → standard cross-entropy
- gamma=2 (default) → hard examples get ~100x more weight than easy ones
- alpha balances positive vs. negative class importance
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss operating on raw logits (numerically stable)."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: weight for the positive (churn) class. (1-alpha) for negative.
            gamma: focusing parameter. Higher → more focus on hard examples.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B,) raw model output (before sigmoid)
            targets: (B,) binary labels {0, 1}
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
