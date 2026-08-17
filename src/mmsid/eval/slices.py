"""Break results down by item popularity, history length and image availability."""

import numpy as np

from mmsid.eval.metrics import summarise

POP_BUCKETS = ("cold", "tail", "mid", "head")
HISTORY_BUCKETS = ("under_5", "5_to_10", "over_10")


def popularity_bucket(pop: np.ndarray) -> np.ndarray:
    """Bucket index per item_idx. cold = no training interaction. Items seen in training are
    ranked by popularity: the bottom 50% are tail, the next 30% mid, the top 20% head."""
    bucket = np.zeros(len(pop), np.int64)
    warm = np.flatnonzero(pop > 0)
    order = warm[np.argsort(pop[warm], kind="stable")]
    n = len(order)
    bucket[order[: int(0.5 * n)]] = 1
    bucket[order[int(0.5 * n) : int(0.8 * n)]] = 2
    bucket[order[int(0.8 * n) :]] = 3
    return bucket


def history_bucket(history_len: np.ndarray) -> np.ndarray:
    return np.where(history_len < 5, 0, np.where(history_len <= 10, 1, 2))


def slice_metrics(ranks: np.ndarray, target: np.ndarray, history_len: np.ndarray,
                  pop: np.ndarray, has_image: np.ndarray, ks=(5, 10)) -> dict:
    """has_image is indexed by item_idx (row 0 = padding)."""
    out = {}
    pb = popularity_bucket(pop)[target]
    for i, name in enumerate(POP_BUCKETS):
        out[f"pop/{name}"] = summarise(ranks[pb == i], ks)
    hb = history_bucket(history_len)
    for i, name in enumerate(HISTORY_BUCKETS):
        out[f"history/{name}"] = summarise(ranks[hb == i], ks)
    img = has_image[target]
    out["image/real"] = summarise(ranks[img], ks)
    out["image/stand_in"] = summarise(ranks[~img], ks)
    return out
