import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def ensure_artifact_dirs(cfg: Dict[str, Any]) -> None:
    """Create directories for checkpoints/reports/figures if they don't exist."""
    for key in ["checkpoints", "reports", "figures"]:
        dir_path = cfg.get("paths", {}).get(key)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

