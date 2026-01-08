import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_cfg: str = "auto") -> torch.device:
    """Return the appropriate torch device."""
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summary_from_config(cfg: Dict[str, Any]) -> str:
    """Human readable summary for logging."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    return (
        f"[Model] {model_cfg.get('type','transformer')} "
        f"(d_model={model_cfg.get('d_model')}, heads={model_cfg.get('n_heads')}, "
        f"layers={model_cfg.get('num_layers')}) | "
        f"[Data] seq_len={data_cfg.get('seq_len')} | "
        f"[Train] epochs={train_cfg.get('num_epochs')} lr={train_cfg.get('lr')}"
    )

