"""
Multi-Scale Temporal Attention Network (MSTAN)
================================================
A novel architecture for customer churn prediction that combines:

1. Multi-Scale Temporal Convolutions — capture short/medium/long-range
   behavioural patterns through parallel dilated causal convolutions.
2. Gated Feature Fusion — learn to weight each temporal scale dynamically
   per sample using a soft gating mechanism.
3. Temporal Decay Attention Bias — inject an inductive bias that recent
   customer interactions matter more than older ones, via a learnable
   exponential decay added to attention logits.
4. Residual CLS-token pooling with the Transformer encoder backbone.

Why is this novel?
- Standard Transformers treat every time-step equally; MSTAN couples
  multi-resolution temporal convolutions with a decay-biased attention.
- The gated fusion lets the model *select* the best temporal resolution
  per customer — someone with a slow contract downgrade benefits from
  long-range conv while someone who suddenly called support benefits
  from short-range.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Causal (left-padded) 1-D convolution so future info never leaks."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class MultiScaleTemporalBlock(nn.Module):
    """Parallel causal convolutions at multiple dilations + gated fusion."""

    def __init__(self, d_model: int, scales: Tuple[int, ...] = (1, 2, 4), kernel_size: int = 3):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(d_model, d_model, kernel_size=kernel_size, dilation=d),
                nn.GELU(),
                nn.BatchNorm1d(d_model),
            )
            for d in scales
        ])
        # Gated fusion: produce a weight per scale per timestep
        self.gate = nn.Sequential(
            nn.Linear(d_model * len(scales), len(scales)),
        )
        self.n_scales = len(scales)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        x_t = x.transpose(1, 2)  # (B, D, T) for conv

        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x_t).transpose(1, 2))  # (B, T, D) each

        # Stack: (B, T, n_scales, D)
        stacked = torch.stack(branch_outs, dim=2)

        # Gate weights from concatenated branch outputs
        concat = torch.cat(branch_outs, dim=-1)  # (B, T, n_scales*D)
        gate_logits = self.gate(concat)  # (B, T, n_scales)
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)  # (B, T, n_scales, 1)

        fused = (stacked * gate_weights).sum(dim=2)  # (B, T, D)
        return self.proj(fused) + x  # residual


class TemporalDecayAttention(nn.Module):
    """
    Multi-head self-attention with a learnable temporal decay bias.

    For positions i (query) and j (key), we add a bias  -alpha * |i - j|
    to the attention logits before softmax, where alpha > 0 is a learned
    per-head parameter.  This softly encourages the model to attend more
    to recent timesteps while still allowing long-range attention.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable decay rate per head (initialized small positive)
        self.decay_alpha = nn.Parameter(torch.full((nhead,), 0.1))

    def _decay_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """(nhead, seq_len, seq_len) decay bias matrix."""
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (T, T)
        alpha = F.softplus(self.decay_alpha)  # ensure positive
        bias = -alpha.view(-1, 1, 1) * dist.unsqueeze(0)  # (H, T, T)
        return bias

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (B, T, D)"""
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dk)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn_logits = attn_logits + self._decay_bias(T, x.device)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        if need_weights:
            return out, attn_weights
        return out, None


class MSTANEncoderLayer(nn.Module):
    """One MSTAN encoder layer = TemporalDecayAttention + FFN with pre-norm."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.attn = TemporalDecayAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm architecture (more stable)
        normed = self.norm1(x)
        attn_out, attn_w = self.attn(normed, need_weights=need_weights)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


# ---------------------------------------------------------------------------
# Full MSTAN Model
# ---------------------------------------------------------------------------

class MSTANChurnClassifier(nn.Module):
    """
    Multi-Scale Temporal Attention Network for Churn Prediction.

    Pipeline:
        Input → Linear projection → Multi-Scale Temporal Conv (gated fusion)
              → Positional Encoding → [CLS] token prepend
              → N × MSTAN Encoder Layers (with temporal decay attention)
              → CLS pooling → Classification head
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        input_dropout: float = 0.1,
        scales: Tuple[int, ...] = (1, 2, 4),
        conv_kernel_size: int = 3,
        max_len: int = 500,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(input_dropout)

        # ---- Novel component 1: Multi-Scale Temporal Convolutions ----
        self.ms_block = MultiScaleTemporalBlock(
            d_model=d_model, scales=scales, kernel_size=conv_kernel_size
        )

        # Positional encoding (sinusoidal)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- Novel component 2: Temporal Decay Attention Layers ----
        self.layers = nn.ModuleList([
            MSTANEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Classification head with an extra hidden layer for capacity
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, input_dim)
            return_attn: whether to return attention weights from last layer
        Returns:
            logits: (B,)
            attn: (B, H, T+1, T+1) or None
        """
        B, T, _ = x.shape

        # Project input features to d_model
        x = self.input_proj(x)
        x = self.input_dropout(x)

        # Multi-scale temporal convolutions with gated fusion
        x = self.ms_block(x)  # (B, T, D)

        # Add positional encoding
        x = x + self.pe[:, :T]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, D)

        # Transformer encoder with temporal decay attention
        attn_weights = None
        for i, layer in enumerate(self.layers):
            need_w = return_attn and (i == len(self.layers) - 1)
            x, attn_weights = layer(x, need_weights=need_w)

        x = self.final_norm(x)

        # CLS token pooling
        cls_out = x[:, 0]
        cls_out = self.dropout(cls_out)
        logits = self.classifier(cls_out).squeeze(-1)

        return logits, attn_weights
