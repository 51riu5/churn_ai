from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import ensure_artifact_dirs, load_config
from src.data.datamodule import TemporalDataModule
from src.training.metrics import classification_metrics
from src.training.utils import get_device, set_seed


def pool_sequences(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Temporal average pooling to create static vectors for sklearn baselines."""
    X = dataset.sequences.mean(dim=1).numpy()
    y = dataset.labels.numpy()
    return X, y


def train_log_reg(train_X, train_y) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=-1,
        solver="liblinear",
    )
    clf.fit(train_X, train_y)
    return clf


def train_random_forest(train_X, train_y) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(train_X, train_y)
    return clf


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Dict:
    probs = model.predict_proba(X)[:, 1]
    return classification_metrics(y_true=y, y_prob=probs)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_artifact_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    data_module = TemporalDataModule(cfg, device=device)
    data_module.setup()
    train_loader, val_loader, test_loader, _ = data_module.dataloaders()

    train_X, train_y = pool_sequences(train_loader.dataset)
    val_X, val_y = pool_sequences(val_loader.dataset)
    test_X, test_y = pool_sequences(test_loader.dataset)

    log_reg = train_log_reg(train_X, train_y)
    rf = train_random_forest(train_X, train_y)

    results = {
        "logistic_regression": {
            "val": evaluate_model(log_reg, val_X, val_y),
            "test": evaluate_model(log_reg, test_X, test_y),
        },
        "random_forest": {
            "val": evaluate_model(rf, val_X, val_y),
            "test": evaluate_model(rf, test_X, test_y),
        },
    }

    report_path = os.path.join(cfg["paths"]["reports"], "baseline_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline metrics to {report_path}")
    for model_name, splits in results.items():
        test_f1 = splits['test']["f1"]
        print(f"{model_name} test F1: {test_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sklearn baselines for churn")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
