"""Popularity baselines: the same list for every user."""

import numpy as np
import pandas as pd

from mmsid.paths import processed_dir


def overall_top(pop: np.ndarray, k: int) -> np.ndarray:
    """The k items with the most training interactions."""
    return np.argsort(-pop, kind="stable")[:k]


def recent_top(data: str, split: str, n_items: int, k: int, window_days: int = 90) -> np.ndarray:
    """The k most frequent training targets in the last `window_days` of the training period."""
    train = pd.read_parquet(processed_dir(data, split) / "train.parquet", columns=["target", "timestamp"])
    ts = train["timestamp"].to_numpy()
    unit = 1000 if ts.max() > 10**11 else 1  # the 2023 data is in milliseconds
    recent = train[ts >= ts.max() - window_days * 86400 * unit]
    counts = np.bincount(recent["target"].to_numpy(), minlength=n_items + 1)
    return np.argsort(-counts, kind="stable")[:k]
