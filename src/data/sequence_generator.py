from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .preprocessing import CATEGORICAL_COLS, NUMERIC_COLS


def _simulate_numeric_sequence(
    value: float, seq_len: int, drift_std: float, positive: bool, rng: np.random.Generator
) -> np.ndarray:
    """Create a simple random-walk style numeric trajectory."""
    base = np.full(seq_len, value, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=abs(value) * drift_std + 1e-4, size=seq_len)
    trajectory = np.cumsum(noise) + base
    if positive:
        trajectory = np.clip(trajectory, a_min=0.0, a_max=None)
    return trajectory


def generate_sequence_for_row(
    row: pd.Series,
    seq_len: int,
    drift_std: float,
    rng: np.random.Generator,
    categorical_cols: List[str] = CATEGORICAL_COLS,
    numeric_cols: List[str] = NUMERIC_COLS,
) -> pd.DataFrame:
    """Expand a single customer record into a temporal sequence."""
    numeric_data = {}
    for col in numeric_cols:
        numeric_data[col] = _simulate_numeric_sequence(
            float(row[col]), seq_len=seq_len, drift_std=drift_std, positive=True, rng=rng
        )

    seq_frames = []
    for t in range(seq_len):
        step_data = {col: numeric_data[col][t] for col in numeric_cols}
        # keep categorical variables fixed over time for stability
        for col in categorical_cols:
            step_data[col] = row[col]
        seq_frames.append(step_data)
    return pd.DataFrame(seq_frames)


def generate_sequences(
    df: pd.DataFrame,
    preprocessor,
    seq_len: int,
    drift_std: float,
    target_col: str,
    id_col: str,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate encoded sequences and labels from a cleaned dataframe."""
    rng = np.random.default_rng(seed=random_state)
    sequences = []
    labels = []

    for _, row in df.iterrows():
        seq_df = generate_sequence_for_row(
            row=row,
            seq_len=seq_len,
            drift_std=drift_std,
            rng=rng,
        )
        encoded = preprocessor.transform(seq_df)
        sequences.append(encoded.astype(np.float32))
        labels.append(float(row[target_col]))

    X = np.stack(sequences)  # (N, T, F)
    y = np.array(labels, dtype=np.float32)
    return X, y

