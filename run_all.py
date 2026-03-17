"""
run_all.py  —  One-click orchestrator
=====================================
Trains baselines (Logistic Regression, Random Forest), deep learning models
(LSTM, Transformer), and the novel MSTAN architecture, then generates a
comprehensive comparative dashboard with publication-quality figures.

Usage:
    python run_all.py                         # full run
    python run_all.py --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.config import ensure_artifact_dirs, load_config
from src.data.datamodule import TemporalDataModule
from src.training.metrics import (
    classification_metrics,
    classification_metrics_optimized,
    optimal_threshold,
)
from src.training.train import build_criterion, build_model, run_epoch
from src.training.early_stopping import EarlyStopping
from src.training.utils import count_parameters, get_device, save_checkpoint, set_seed
from src.evaluation.visualize import (
    plot_confusion_matrix,
    plot_model_comparison_bar,
    plot_multi_pr,
    plot_multi_roc,
    plot_radar_chart,
    plot_training_history,
)
from src.training.metrics import confusion_matrix


def _separator(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ── Sklearn baselines ──────────────────────────────────────────────

def pool_sequences(dataset) -> tuple:
    X = dataset.sequences.mean(dim=1).numpy()
    y = dataset.labels.numpy()
    return X, y


def train_sklearn_baselines(
    train_ds, val_ds, test_ds
) -> Dict[str, Dict]:
    """Train LogReg, RF, and GBM baselines and return results."""
    train_X, train_y = pool_sequences(train_ds)
    val_X, val_y = pool_sequences(val_ds)
    test_X, test_y = pool_sequences(test_ds)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")
    lr.fit(train_X, train_y)
    lr_probs_val = lr.predict_proba(val_X)[:, 1]
    lr_probs_test = lr.predict_proba(test_X)[:, 1]
    lr_thresh = optimal_threshold(val_y, lr_probs_val)
    results["LogReg"] = {
        "metrics": classification_metrics(test_y, lr_probs_test, threshold=lr_thresh),
        "y_true": test_y, "y_prob": lr_probs_test,
    }

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf.fit(train_X, train_y)
    rf_probs_val = rf.predict_proba(val_X)[:, 1]
    rf_probs_test = rf.predict_proba(test_X)[:, 1]
    rf_thresh = optimal_threshold(val_y, rf_probs_val)
    results["RF"] = {
        "metrics": classification_metrics(test_y, rf_probs_test, threshold=rf_thresh),
        "y_true": test_y, "y_prob": rf_probs_test,
    }

    return results


# ── Deep learning training ──────────────────────────────────────────

def train_dl_model(
    model_type: str, cfg: Dict, device: torch.device,
    train_loader, val_loader, test_loader, feature_dim: int,
) -> Dict:
    """Train a single deep learning model and return results + history."""
    model = build_model(model_type, feature_dim, cfg).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    train_cfg = cfg["training"]
    criterion = build_criterion(cfg)

    if train_cfg.get("optimizer", "adamw").lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg["lr"],
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

    early_stopper = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 10))
    best_val_loss = float("inf")
    ckpt_path = os.path.join(cfg["paths"]["checkpoints"], f"best_{model_type}.pt")

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "val_auc": []}

    for epoch in range(train_cfg["num_epochs"]):
        train_loss, train_met, _, _ = run_epoch(
            model, train_loader, criterion, device, train=True,
            optimizer=optimizer, grad_clip=train_cfg.get("grad_clip_norm", 1.0),
        )
        val_loss, val_met, _, _ = run_epoch(
            model, val_loader, criterion, device, train=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_met["f1"])
        history["val_f1"].append(val_met["f1"])
        history["val_auc"].append(val_met["roc_auc"])

        if scheduler:
            scheduler.step()

        print(
            f"    Epoch {epoch+1:02d} | "
            f"Train L={train_loss:.4f} F1={train_met['f1']:.3f} | "
            f"Val L={val_loss:.4f} F1={val_met['f1']:.3f} AUC={val_met['roc_auc']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, ckpt_path)

        if early_stopper.step(val_loss):
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Evaluate best model
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    _, _, val_labels, val_probs = run_epoch(
        model, val_loader, criterion, device, train=False,
    )
    best_thresh = optimal_threshold(val_labels, val_probs)

    _, _, test_labels, test_probs = run_epoch(
        model, test_loader, criterion, device, train=False,
    )
    test_met = classification_metrics(test_labels, test_probs, threshold=best_thresh)

    return {
        "metrics": test_met,
        "y_true": test_labels,
        "y_prob": test_probs,
        "history": history,
    }


# ── Main orchestrator ───────────────────────────────────────────────

def main(config_path: str = "configs/base_config.yaml") -> None:
    cfg = load_config(config_path)
    ensure_artifact_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    _separator("LOADING DATA")
    t0 = time.time()
    dm = TemporalDataModule(cfg, device=device)
    dm.setup()
    train_loader, val_loader, test_loader, feature_dim = dm.dataloaders()
    print(f"  Feature dim: {feature_dim}  |  Seq len: {cfg['data']['seq_len']}")
    print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")
    print(f"  Data loaded in {time.time()-t0:.1f}s")

    # Save preprocessor for the dashboard API (fast model loading)
    pkl_path = os.path.join(cfg["paths"]["checkpoints"], "preprocessor.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(dm.preprocessor, f)
    print(f"  Saved preprocessor -> {pkl_path}")

    all_results: Dict[str, Dict] = {}
    histories: Dict[str, Dict] = {}

    # ── Baselines ────────────────────────────────────────────────
    _separator("SKLEARN BASELINES")
    t0 = time.time()
    baseline_results = train_sklearn_baselines(
        train_loader.dataset, val_loader.dataset, test_loader.dataset
    )
    for name, res in baseline_results.items():
        all_results[name] = res
        m = res["metrics"]
        print(f"  {name:15s} | F1={m['f1']:.3f}  Rec={m['recall']:.3f}  "
              f"AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  t={m['threshold']:.2f}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Deep Learning Models ─────────────────────────────────────
    dl_models = ["lstm", "transformer", "mstan"]
    dl_display = {"lstm": "LSTM", "transformer": "Transformer", "mstan": "MSTAN (Ours)"}

    for model_type in dl_models:
        display_name = dl_display[model_type]
        _separator(f"TRAINING {display_name.upper()}")
        t0 = time.time()
        res = train_dl_model(
            model_type, cfg, device,
            train_loader, val_loader, test_loader, feature_dim,
        )
        all_results[display_name] = res
        histories[display_name] = res["history"]
        m = res["metrics"]
        elapsed = time.time() - t0
        print(f"\n  {display_name} Test | F1={m['f1']:.3f}  Rec={m['recall']:.3f}  "
              f"Prec={m['precision']:.3f}  AUC={m['roc_auc']:.3f}  "
              f"PR-AUC={m['pr_auc']:.3f}  t={m['threshold']:.2f}  ({elapsed:.1f}s)")

    # ── Save reports ─────────────────────────────────────────────
    _separator("SAVING REPORTS & FIGURES")
    report_dir = cfg["paths"]["reports"]
    fig_dir = cfg["paths"]["figures"]

    # Consolidated JSON report
    consolidated = {}
    for name, res in all_results.items():
        consolidated[name] = res["metrics"]
    with open(os.path.join(report_dir, "all_models_comparison.json"), "w") as f:
        json.dump(consolidated, f, indent=2)

    # Per-model JSON
    for name, res in all_results.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        with open(os.path.join(report_dir, f"{safe_name}_metrics.json"), "w") as f:
            json.dump(res["metrics"], f, indent=2)

    # Save training histories for interactive dashboard charts
    with open(os.path.join(report_dir, "training_histories.json"), "w") as f:
        json.dump(histories, f, indent=2)

    # ── Generate dashboard figures ───────────────────────────────
    metric_dict = {name: res["metrics"] for name, res in all_results.items()}
    curve_dict = {name: {"y_true": res["y_true"], "y_prob": res["y_prob"]}
                  for name, res in all_results.items()}

    plot_model_comparison_bar(metric_dict, os.path.join(fig_dir, "comparison_bar.png"))
    plot_multi_roc(curve_dict, os.path.join(fig_dir, "multi_roc.png"))
    plot_multi_pr(curve_dict, os.path.join(fig_dir, "multi_pr.png"))
    plot_radar_chart(metric_dict, os.path.join(fig_dir, "radar_chart.png"))

    if histories:
        plot_training_history(histories, os.path.join(fig_dir, "training_history.png"))

    # Per-model confusion matrices
    for name, res in all_results.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        thresh = res["metrics"]["threshold"]
        y_pred = (res["y_prob"] >= thresh).astype(int)
        cm = confusion_matrix(res["y_true"], y_pred)
        plot_confusion_matrix(cm, ["No Churn", "Churn"],
                              os.path.join(fig_dir, f"{safe_name}_confusion.png"))

    # ── Final summary table ──────────────────────────────────────
    _separator("FINAL RESULTS SUMMARY")
    header = f"  {'Model':20s} {'F1':>6s} {'Recall':>8s} {'Prec':>8s} {'AUC':>6s} {'PR-AUC':>7s} {'Thresh':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in ["LogReg", "RF", "LSTM", "Transformer", "MSTAN (Ours)"]:
        if name in all_results:
            m = all_results[name]["metrics"]
            marker = " <-- NOVEL" if "MSTAN" in name else ""
            print(f"  {name:20s} {m['f1']:6.3f} {m['recall']:8.3f} {m['precision']:8.3f} "
                  f"{m['roc_auc']:6.3f} {m['pr_auc']:7.3f} {m['threshold']:7.2f}{marker}")

    print(f"\n  Reports -> {report_dir}/")
    print(f"  Figures -> {fig_dir}/")
    print(f"\n  Key figures:")
    print(f"    - comparison_bar.png    (grouped bar chart)")
    print(f"    - multi_roc.png         (overlaid ROC curves)")
    print(f"    - multi_pr.png          (overlaid PR curves)")
    print(f"    - radar_chart.png       (spider chart)")
    print(f"    - training_history.png  (loss/F1/AUC curves)")
    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
