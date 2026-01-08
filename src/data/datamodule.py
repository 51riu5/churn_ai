from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    build_preprocessor,
    clean_telco_data,
    fit_preprocessor,
    validate_telco_columns,
)
from .sequence_generator import generate_sequences


class SequenceDataset(Dataset):
    """Torch dataset wrapping temporal feature tensors and labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


@dataclass
class DataSplits:
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    feature_dim: int


class TemporalDataModule:
    """End-to-end data module: load CSV, preprocess, build sequences, and dataloaders."""

    def __init__(self, cfg: Dict, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.data_cfg = cfg["data"]
        self.seq_len = self.data_cfg.get("seq_len", 12)
        self.drift_std = self.data_cfg.get("drift_std", 0.03)
        self.batch_size = cfg["training"]["batch_size"]
        self.target_col = self.data_cfg.get("target_col", "Churn")
        self.id_col = self.data_cfg.get("id_col", "customerID")
        self.random_state = self.data_cfg.get("random_state", 42)

        self.preprocessor = build_preprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
        self.splits: Optional[DataSplits] = None

    def _load_dataframe(self) -> pd.DataFrame:
        path = self.data_cfg["path"]
        df = pd.read_csv(path)
        validate_telco_columns(df, target_col=self.target_col)
        df = clean_telco_data(df, target_col=self.target_col)
        return df

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_size = self.data_cfg.get("test_size", 0.15)
        val_size = self.data_cfg.get("val_size", 0.15)

        train_df, temp_df = train_test_split(
            df,
            test_size=test_size + val_size,
            random_state=self.random_state,
            stratify=df[self.target_col],
        )
        val_rel = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_rel,
            random_state=self.random_state,
            stratify=temp_df[self.target_col],
        )
        return train_df, val_df, test_df

    def setup(self) -> None:
        df = self._load_dataframe()
        train_df, val_df, test_df = self._split(df)

        # Fit preprocessing on train only to avoid leakage
        self.preprocessor = fit_preprocessor(
            self.preprocessor, train_df, target_col=self.target_col, id_col=self.id_col
        )

        train_X, train_y = generate_sequences(
            train_df,
            preprocessor=self.preprocessor,
            seq_len=self.seq_len,
            drift_std=self.drift_std,
            target_col=self.target_col,
            id_col=self.id_col,
            random_state=self.random_state,
        )
        val_X, val_y = generate_sequences(
            val_df,
            preprocessor=self.preprocessor,
            seq_len=self.seq_len,
            drift_std=self.drift_std,
            target_col=self.target_col,
            id_col=self.id_col,
            random_state=self.random_state + 1,
        )
        test_X, test_y = generate_sequences(
            test_df,
            preprocessor=self.preprocessor,
            seq_len=self.seq_len,
            drift_std=self.drift_std,
            target_col=self.target_col,
            id_col=self.id_col,
            random_state=self.random_state + 2,
        )

        feature_dim = train_X.shape[-1]
        self.splits = DataSplits(
            train=SequenceDataset(train_X, train_y),
            val=SequenceDataset(val_X, val_y),
            test=SequenceDataset(test_X, test_y),
            feature_dim=feature_dim,
        )

    def dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        if self.splits is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        train_loader = DataLoader(
            self.splits.train, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            self.splits.val, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        test_loader = DataLoader(
            self.splits.test, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        return train_loader, val_loader, test_loader, self.splits.feature_dim

