from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayerWithAttention(nn.Module):
    """Transformer encoder layer that can optionally return attention weights."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self, src: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output, attn_weights = self.self_attn(
            src, src, src, need_weights=need_weights, average_attn_weights=False
        )
        src = self.norm1(src + self.dropout1(attn_output))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src, attn_weights if need_weights else None


class TransformerChurnClassifier(nn.Module):
    """Transformer encoder classifier for churn prediction."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "mean",
        use_cls_token: bool = True,
        input_dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(input_dropout)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttention(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.pooling = pooling
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            return_attn: if True, returns attention weights from last layer
        Returns:
            logits: (batch, 1)
            attn_weights: (batch, heads, seq_len(+1), seq_len(+1)) or None
        """
        x = self.input_proj(x)
        x = self.input_dropout(x)
        x = self.pos_encoding(x)

        attn_weights = None
        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)

        for i, layer in enumerate(self.layers):
            need_weights = return_attn and (i == len(self.layers) - 1)
            x, attn_weights = layer(x, need_weights=need_weights)

        if self.pooling == "cls" and self.use_cls_token:
            pooled = x[:, 0, :]
        elif self.pooling == "max":
            pooled, _ = torch.max(x, dim=1)
        else:
            pooled = torch.mean(x, dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, attn_weights

