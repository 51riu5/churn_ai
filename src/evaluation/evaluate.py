from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch

from src.config import ensure_artifact_dirs, load_config
from src.data.datamodule import TemporalDataModule
from src.evaluation.interpretability import extract_attention
from src.evaluation.visualize import plot_attention_weights, plot_confusion_matrix, plot_roc
from src.models.lstm import LSTMChurnClassifier
from src.models.transformer import TransformerChurnClassifier
from src.training.metrics import classification_metrics, confusion_matrix
from src.training.utils import get_device, set_seed


def build_model(model_type: str, input_dim: int, cfg: Dict) -> torch.nn.Module:
    model_cfg = cfg["model"]
    if model_type == "transformer":
        return TransformerChurnClassifier(
            input_dim=input_dim,
            d_model=model_cfg.get("d_model", 64),
            nhead=model_cfg.get("n_heads", 4),
            num_layers=model_cfg.get("num_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 128),
            dropout=model_cfg.get("dropout", 0.1),
            pooling=model_cfg.get("pooling", "mean"),
            use_cls_token=model_cfg.get("use_cls_token", True),
            input_dropout=model_cfg.get("input_dropout", 0.1),
        )
    elif model_type == "lstm":
        return LSTMChurnClassifier(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("d_model", 64),
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def evaluate(model, loader, device) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch_y.numpy()
            all_probs.append(probs)
            all_labels.append(labels)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    mets = classification_metrics(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return mets, cm, y_true, y_prob


def main(config_path: str, checkpoint: str, model_type: str) -> None:
    cfg = load_config(config_path)
    ensure_artifact_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    data_module = TemporalDataModule(cfg, device=device)
    data_module.setup()
    _, _, test_loader, feature_dim = data_module.dataloaders()

    model = build_model(model_type=model_type, input_dim=feature_dim, cfg=cfg).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    metrics, cm, y_true, y_prob = evaluate(model, test_loader, device)

    report_path = os.path.join(cfg["paths"]["reports"], f"{model_type}_test_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {report_path}")

    fig_dir = cfg["paths"]["figures"]
    os.makedirs(fig_dir, exist_ok=True)
    plot_confusion_matrix(
        cm, class_names=["No", "Yes"], path=os.path.join(fig_dir, f"{model_type}_confusion.png")
    )
    plot_roc(
        y_true=y_true,
        y_prob=y_prob,
        path=os.path.join(fig_dir, f"{model_type}_roc.png"),
    )

    # Interpretability for Transformer
    if model_type == "transformer":
        batch_x, _ = next(iter(test_loader))
        attn = extract_attention(model, batch_x, device=device)
        if attn is not None:
            attn_np = attn.cpu().numpy()
            plot_attention_weights(
                attn_np,
                path=os.path.join(fig_dir, f"{model_type}_attention.png"),
                title="Mean attention (last layer)",
            )

    print("Test metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate churn model checkpoint")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="transformer", choices=["transformer", "lstm"]
    )
    args = parser.parse_args()
    main(config_path=args.config, checkpoint=args.checkpoint, model_type=args.model)
