"""
Churn-Correlated Temporal Sequence Generator
=============================================
Instead of naive random walks, this generator creates *realistic* temporal
behaviour patterns that correlate with the churn label:

Churners:
  - tenure: stagnant or slightly declining perceived value
  - MonthlyCharges: tend to increase (price hikes push churn)
  - TotalCharges: slower growth (disengagement)
  - Some categorical features may flip mid-sequence (e.g. drop services)

Non-churners:
  - tenure: stable baseline
  - MonthlyCharges: stable or slightly decreasing (loyalty discounts)
  - TotalCharges: steady growth

This makes the synthetic temporal data *informative* so that sequence models
can actually learn temporal patterns rather than fitting noise.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .preprocessing import CATEGORICAL_COLS, NUMERIC_COLS


def _simulate_numeric_sequence(
    value: float,
    seq_len: int,
    drift_std: float,
    positive: bool,
    rng: np.random.Generator,
    trend: float = 0.0,
) -> np.ndarray:
    """Create a trajectory with optional trend + noise.

    Args:
        value: final observed value (placed at last timestep).
        seq_len: number of timesteps.
        drift_std: noise standard deviation (relative to value).
        positive: clip to >=0.
        rng: random generator.
        trend: per-step linear trend added to the trajectory.
    """
    # Build trajectory backwards from the observed value
    steps = np.arange(seq_len, dtype=np.float32)
    base = value - trend * (seq_len - 1 - steps)  # linear trend towards value
    noise = rng.normal(loc=0.0, scale=abs(value) * drift_std + 1e-4, size=seq_len)
    trajectory = base + noise
    if positive:
        trajectory = np.clip(trajectory, a_min=0.0, a_max=None)
    return trajectory.astype(np.float32)


def _churn_trends(is_churn: bool, rng: np.random.Generator) -> dict:
    """Return per-column trend parameters conditioned on churn label."""
    if is_churn:
        return {
            "tenure": rng.uniform(-0.05, 0.02),       # stagnating
            "MonthlyCharges": rng.uniform(0.1, 0.5),   # increasing charges
            "TotalCharges": rng.uniform(-0.5, 0.3),     # slow growth
        }
    else:
        return {
            "tenure": rng.uniform(0.0, 0.1),           # stable growth
            "MonthlyCharges": rng.uniform(-0.2, 0.1),  # stable/decreasing
            "TotalCharges": rng.uniform(0.2, 1.0),      # healthy growth
        }


def _maybe_flip_categorical(
    row: pd.Series,
    col: str,
    seq_len: int,
    is_churn: bool,
    rng: np.random.Generator,
) -> List[str]:
    """For churners, maybe simulate a service downgrade mid-sequence."""
    val = row[col]
    values = [val] * seq_len

    if not is_churn:
        return values

    # Service-related columns may see a downgrade for churners
    service_cols = {
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    }
    if col in service_cols and val == "Yes" and rng.random() < 0.3:
        flip_point = rng.integers(seq_len // 3, seq_len - 1)
        for t in range(flip_point, seq_len):
            values[t] = "No"

    return values


def generate_sequence_for_row(
    row: pd.Series,
    seq_len: int,
    drift_std: float,
    rng: np.random.Generator,
    is_churn: bool,
    categorical_cols: List[str] = CATEGORICAL_COLS,
    numeric_cols: List[str] = NUMERIC_COLS,
) -> pd.DataFrame:
    """Expand a single customer record into a temporal sequence with churn-correlated patterns."""
    trends = _churn_trends(is_churn, rng)

    numeric_data = {}
    for col in numeric_cols:
        trend = trends.get(col, 0.0)
        numeric_data[col] = _simulate_numeric_sequence(
            float(row[col]),
            seq_len=seq_len,
            drift_std=drift_std,
            positive=True,
            rng=rng,
            trend=trend,
        )

    seq_frames = []
    # Pre-compute categorical sequences
    cat_sequences = {}
    for col in categorical_cols:
        cat_sequences[col] = _maybe_flip_categorical(row, col, seq_len, is_churn, rng)

    for t in range(seq_len):
        step_data = {col: numeric_data[col][t] for col in numeric_cols}
        for col in categorical_cols:
            step_data[col] = cat_sequences[col][t]
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
        is_churn = bool(row[target_col] == 1)
        seq_df = generate_sequence_for_row(
            row=row,
            seq_len=seq_len,
            drift_std=drift_std,
            rng=rng,
            is_churn=is_churn,
        )
        encoded = preprocessor.transform(seq_df)
        sequences.append(encoded.astype(np.float32))
        labels.append(float(row[target_col]))

    X = np.stack(sequences)  # (N, T, F)
    y = np.array(labels, dtype=np.float32)
    return X, y
