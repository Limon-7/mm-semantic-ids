"""Turn content codes into unique item IDs by appending a collision token."""

import numpy as np
import pandas as pd


def add_collision_token(codes: np.ndarray) -> np.ndarray:
    """codes: [n_items, levels]. Returns [n_items, levels + 1].

    Items that share all content codes are numbered 0, 1, 2, ... in item order, as in TIGER.
    """
    df = pd.DataFrame(codes)
    collision = df.groupby(list(df.columns), sort=False).cumcount().to_numpy()
    return np.concatenate([codes, collision[:, None]], axis=1).astype(np.int64)


def with_padding_row(ids: np.ndarray) -> np.ndarray:
    """Prepend a row of zeros so the array can be indexed by item_idx (0 is padding)."""
    return np.concatenate([np.zeros((1, ids.shape[1]), ids.dtype), ids])


def level_sizes(ids: np.ndarray, codebook_sizes: list[int]) -> list[int]:
    """Vocabulary size of every position: the content codebooks, then the collision level."""
    return [*codebook_sizes, int(ids[:, -1].max()) + 1]
