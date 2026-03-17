from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ensure_artifact_dirs, load_config
from src.data.datamodule import TemporalDataModule
from src.models.lstm import LSTMChurnClassifier
from src.models.mstan import MSTANChurnClassifier
from src.models.transformer import TransformerChurnClassifier
from src.training.early_stopping import EarlyStopping
from src.training.focal_loss import FocalLoss
from src.training.metrics import (
    classification_metrics,
    classification_metrics_optimized,
    optimal_threshold,
)
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
    elif model_type == "mstan":
        return MSTANChurnClassifier(
            input_dim=input_dim,
            d_model=model_cfg.get("d_model", 64),
            nhead=model_cfg.get("n_heads", 4),
            num_layers=model_cfg.get("num_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 128),
            dropout=model_cfg.get("dropout", 0.1),
            input_dropout=model_cfg.get("input_dropout", 0.1),
            scales=tuple(model_cfg.get("scales", [1, 2, 4])),
            conv_kernel_size=model_cfg.get("conv_kernel_size", 3),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def build_criterion(cfg: Dict) -> nn.Module:
    """Build loss function based on config."""
    loss_type = cfg.get("training", {}).get("loss", "focal")
    if loss_type == "focal":
        alpha = cfg.get("training", {}).get("focal_alpha", 0.75)
        gamma = cfg.get("training", {}).get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    return nn.BCEWithLogitsLoss()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
    optimizer=None,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
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
    met = classification_metrics(all_labels, all_probs)
    met["loss"] = avg_loss
    return avg_loss, met, all_labels, all_probs


def main(config_path: str, model_type: str) -> Dict:
    cfg = load_config(config_path)
    ensure_artifact_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    print(summary_from_config(cfg))
    data_module = TemporalDataModule(cfg, device=device)
    data_module.setup()
    train_loader, val_loader, test_loader, feature_dim = data_module.dataloaders()

    model = build_model(model_type=model_type, input_dim=feature_dim, cfg=cfg).to(device)
    print(f"[{model_type.upper()}] Parameters: {count_parameters(model):,}")

    train_cfg = cfg["training"]
    criterion = build_criterion(cfg)
    print(f"  Loss: {criterion.__class__.__name__}")

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

    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    total_epochs = train_cfg["num_epochs"]

    scheduler = None
    if train_cfg.get("scheduler", "none") == "cosine":
        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs
            )

    early_stopper = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 7))
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(
        cfg["paths"]["checkpoints"], f"best_{model_type}.pt"
    )

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "val_auc": []}

    for epoch in range(train_cfg["num_epochs"]):
        train_loss, train_met, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            train=True,
            optimizer=optimizer,
            grad_clip=train_cfg.get("grad_clip_norm", 1.0),
        )
        val_loss, val_met, val_labels, val_probs = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            train=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_met["f1"])
        history["val_f1"].append(val_met["f1"])
        history["val_auc"].append(val_met["roc_auc"])

        if scheduler:
            scheduler.step()

        print(
            f"  Epoch {epoch+1:02d}/{train_cfg['num_epochs']} "
            f"| Train loss {train_loss:.4f} F1 {train_met['f1']:.3f} "
            f"| Val loss {val_loss:.4f} F1 {val_met['f1']:.3f} "
            f"AUC {val_met['roc_auc']:.3f} PR-AUC {val_met['pr_auc']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, checkpoint_path)
            print(f"    -> Saved best model")

        if early_stopper.step(val_loss):
            print("  Early stopping triggered.")
            break

    # Evaluate best model on test
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    _, test_met_default, test_labels, test_probs = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        train=False,
    )

    # Also compute with optimized threshold (found on val set)
    _, _, val_labels_final, val_probs_final = run_epoch(
        model=model, loader=val_loader, criterion=criterion, device=device, train=False,
    )
    best_threshold = optimal_threshold(val_labels_final, val_probs_final)
    test_met_opt = classification_metrics(test_labels, test_probs, threshold=best_threshold)

    print(f"\n  Test (t=0.50) | acc {test_met_default['accuracy']:.3f} "
          f"prec {test_met_default['precision']:.3f} "
          f"rec {test_met_default['recall']:.3f} "
          f"f1 {test_met_default['f1']:.3f} "
          f"auc {test_met_default['roc_auc']:.3f}")
    print(f"  Test (t={best_threshold:.2f}) | acc {test_met_opt['accuracy']:.3f} "
          f"prec {test_met_opt['precision']:.3f} "
          f"rec {test_met_opt['recall']:.3f} "
          f"f1 {test_met_opt['f1']:.3f} "
          f"auc {test_met_opt['roc_auc']:.3f}")

    # Save report
    report = {
        "default_threshold": test_met_default,
        "optimized_threshold": test_met_opt,
        "best_threshold": best_threshold,
        "history": history,
    }
    report_path = os.path.join(cfg["paths"]["reports"], f"{model_type}_test_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report -> {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="mstan",
        choices=["transformer", "lstm", "mstan"],
    )
    args = parser.parse_args()
    main(config_path=args.config, model_type=args.model)
