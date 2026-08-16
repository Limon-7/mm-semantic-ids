"""Load processed splits as dense arrays.

The datasets are small enough to sit on the GPU whole, so batches are made by indexing tensors
instead of going through a DataLoader.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from mmsid.data.preprocess import MAX_HISTORY
from mmsid.paths import processed_dir


@dataclass
class SplitPart:
    user: torch.Tensor      # [n]
    history: torch.Tensor   # [n, MAX_HISTORY], left-padded with 0 so the newest item is last
    target: torch.Tensor    # [n]

    def __len__(self) -> int:
        return len(self.target)

    def to(self, device) -> "SplitPart":
        return SplitPart(self.user.to(device), self.history.to(device), self.target.to(device))

    def subset(self, idx: torch.Tensor) -> "SplitPart":
        return SplitPart(self.user[idx], self.history[idx], self.target[idx])

    @property
    def history_len(self) -> torch.Tensor:
        return (self.history > 0).sum(1)


def trim(history: torch.Tensor) -> torch.Tensor:
    """Drop the left columns that are padding for every row of the batch."""
    longest = int((history > 0).sum(1).max())
    return history[:, history.shape[1] - max(1, longest):]


class BucketSampler:
    """Draws batches of rows with similar history length, so that most batches can be trimmed
    to a few items instead of carrying 20 items of padding. A bucket is picked with probability
    proportional to its size, so every row is still equally likely to be drawn."""

    def __init__(self, history_len: torch.Tensor, edges=(2, 4, 6, 10, 20), seed: int = 0):
        self.gen = torch.Generator(device=history_len.device).manual_seed(seed)
        self.buckets, lo = [], 0
        for hi in edges:
            idx = ((history_len > lo) & (history_len <= hi)).nonzero().flatten()
            if len(idx):
                self.buckets.append(idx)
            lo = hi
        self.weights = torch.tensor([float(len(b)) for b in self.buckets], device=history_len.device)

    def sample(self, batch_size: int) -> torch.Tensor:
        bucket = self.buckets[int(torch.multinomial(self.weights, 1, generator=self.gen))]
        pick = torch.randint(len(bucket), (batch_size,), device=bucket.device, generator=self.gen)
        return bucket[pick]


def pad_histories(histories, max_len: int = MAX_HISTORY) -> np.ndarray:
    out = np.zeros((len(histories), max_len), np.int64)
    for i, h in enumerate(histories):
        h = h[-max_len:]
        out[i, max_len - len(h):] = h
    return out


def load_part(data: str, split: str, part: str) -> SplitPart:
    df = pd.read_parquet(processed_dir(data, split) / f"{part}.parquet")
    return SplitPart(
        torch.from_numpy(df["user_idx"].to_numpy(np.int64)),
        torch.from_numpy(pad_histories(df["history"].tolist())),
        torch.from_numpy(df["target"].to_numpy(np.int64)),
    )


def load_item_pop(data: str, split: str) -> np.ndarray:
    return np.load(processed_dir(data, split) / "item_pop.npy")


def n_items(data: str) -> int:
    return len(pd.read_parquet(processed_dir(data) / "items.parquet", columns=["item_idx"]))
