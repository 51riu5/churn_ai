from __future__ import annotations

from typing import Dict, Optional

import torch


def extract_attention(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Run a forward pass with attention weights returned (Transformer only)."""
    model.eval()
    batch_x = batch_x.to(device)
    with torch.no_grad():
        _, attn = model(batch_x, return_attn=True)  # type: ignore[arg-type]
    return attn


def summarize_attention(attn: torch.Tensor) -> Dict[str, float]:
    """Simple statistics over attention maps for quick sanity checks."""
    if attn is None:
        return {}
    stats = {
        "mean": attn.mean().item(),
        "std": attn.std().item(),
        "max": attn.max().item(),
        "min": attn.min().item(),
    }
    return stats

