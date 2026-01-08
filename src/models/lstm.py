from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class LSTMChurnClassifier(nn.Module):
    """Many-to-one LSTM baseline for sequence modeling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x = self.input_proj(x)
        outputs, _ = self.lstm(x)
        # use last time step representation
        last_hidden = outputs[:, -1, :]
        logits = self.fc(self.dropout(last_hidden)).squeeze(-1)
        return logits, None

