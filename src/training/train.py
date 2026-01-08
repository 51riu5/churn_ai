from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ensure_artifact_dirs, load_config
from src.data.datamodule import TemporalDataModule
from src.models.lstm import LSTMChurnClassifier
from src.models.transformer import TransformerChurnClassifier
from src.training.early_stopping import EarlyStopping
from src.training.metrics import classification_metrics
from src.training.utils import (
    count_parameters,
    get_device,
    save_checkpoint,
    set_seed,
    summary_from_config,
)


def build_model(model_type: str, input_dim: int, cfg: Dict) -> nn.Module:
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
    optimizer=None,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    all_probs = []
    all_labels = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        with torch.set_grad_enabled(train):
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch_y.detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
        epoch_loss += loss.item() * batch_x.size(0)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss = epoch_loss / len(loader.dataset)
    metrics = classification_metrics(all_labels, all_probs)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def main(config_path: str, model_type: str) -> None:
    cfg = load_config(config_path)
    ensure_artifact_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    print(summary_from_config(cfg))
    data_module = TemporalDataModule(cfg, device=device)
    data_module.setup()
    train_loader, val_loader, test_loader, feature_dim = data_module.dataloaders()

    model = build_model(model_type=model_type, input_dim=feature_dim, cfg=cfg).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    train_cfg = cfg["training"]
    criterion = nn.BCEWithLogitsLoss()

    if train_cfg.get("optimizer", "adamw").lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

    scheduler = None
    if train_cfg.get("scheduler", "none") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["num_epochs"]
        )

    early_stopper = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 5))
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(
        cfg["paths"]["checkpoints"], f"best_{model_type}.pt"
    )

    for epoch in range(train_cfg["num_epochs"]):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            train=True,
            optimizer=optimizer,
            grad_clip=train_cfg.get("grad_clip_norm", 1.0),
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            train=False,
        )

        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{train_cfg['num_epochs']} "
            f"| Train loss {train_loss:.4f} F1 {train_metrics['f1']:.3f} "
            f"| Val loss {val_loss:.4f} F1 {val_metrics['f1']:.3f} AUC {val_metrics['roc_auc']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, checkpoint_path)
            print(f"  Saved new best model to {checkpoint_path}")

        if early_stopper.step(val_loss):
            print("Early stopping triggered.")
            break

    # Evaluate best model on test
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    _, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        train=False,
    )
    print(
        f"Test | loss {test_metrics['loss']:.4f} "
        f"acc {test_metrics['accuracy']:.3f} "
        f"precision {test_metrics['precision']:.3f} "
        f"recall {test_metrics['recall']:.3f} "
        f"f1 {test_metrics['f1']:.3f} "
        f"auc {test_metrics['roc_auc']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["transformer", "lstm"],
    )
    args = parser.parse_args()
    main(config_path=args.config, model_type=args.model)
