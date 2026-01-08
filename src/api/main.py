from __future__ import annotations

import os
from typing import List, Dict, Any

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import ensure_artifact_dirs, load_config
from src.data.datamodule import TemporalDataModule
from src.models.transformer import TransformerChurnClassifier
from src.training.utils import get_device, set_seed

app = FastAPI(title="Churn Predictor API", version="1.0")


class ChurnRequest(BaseModel):
    sequence: List[Dict[str, Any]]  # each timestep is a dict of feature: value


def _load_artifacts():
    config_path = os.getenv("CHURN_CONFIG", "configs/base_config.yaml")
    checkpoint_path = os.getenv("CHURN_CHECKPOINT", "artifacts/checkpoints/best_transformer.pt")

    cfg = load_config(config_path)
    ensure_artifact_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    data_module = TemporalDataModule(cfg, device=device)
    data_module.setup()
    _, _, _, feature_dim = data_module.dataloaders()

    model_cfg = cfg["model"]
    model = TransformerChurnClassifier(
        input_dim=feature_dim,
        d_model=model_cfg.get("d_model", 64),
        nhead=model_cfg.get("n_heads", 4),
        num_layers=model_cfg.get("num_layers", 2),
        dim_feedforward=model_cfg.get("dim_feedforward", 128),
        dropout=model_cfg.get("dropout", 0.1),
        pooling=model_cfg.get("pooling", "mean"),
        use_cls_token=model_cfg.get("use_cls_token", True),
        input_dropout=model_cfg.get("input_dropout", 0.1),
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return cfg, device, data_module, model


cfg, device, data_module, model = _load_artifacts()


@app.post("/predict-churn")
def predict_churn(payload: ChurnRequest):
    try:
        df = pd.DataFrame(payload.sequence)
        encoded = data_module.preprocessor.transform(df)  # type: ignore[attr-defined]
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(tensor)
            prob = torch.sigmoid(logits).item()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to score input: {e}")

    risk = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"
    return {"churn_probability": prob, "risk": risk}

