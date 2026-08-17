"""Ranking metrics. Everything is computed from the 0-based rank of the true item in the
recommended list (-1 if it is not in the list), one value per test row."""

import numpy as np


def target_ranks(recommended: np.ndarray, target: np.ndarray) -> np.ndarray:
    """recommended: [n, K] item ids, best first. target: [n]. Returns [n] ranks, -1 = miss."""
    hit = recommended == target[:, None]
    return np.where(hit.any(1), hit.argmax(1), -1)


def recall_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    return ((ranks >= 0) & (ranks < k)).astype(np.float64)


def ndcg_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """With one relevant item the ideal DCG is 1, so NDCG is 1 / log2(rank + 2)."""
    hit = (ranks >= 0) & (ranks < k)
    return np.where(hit, 1.0 / np.log2(np.maximum(ranks, 0) + 2.0), 0.0)


def summarise(ranks: np.ndarray, ks=(5, 10)) -> dict:
    out = {"n": int(len(ranks))}
    for k in ks:
        out[f"recall@{k}"] = float(recall_at_k(ranks, k).mean()) if len(ranks) else None
        out[f"ndcg@{k}"] = float(ndcg_at_k(ranks, k).mean()) if len(ranks) else None
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap over test rows for the mean of (a - b), where a and b are per-row metric values
    of two models on the same rows."""
    rng = np.random.default_rng(seed)
    diff = a - b
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = diff[rng.integers(0, len(diff), len(diff))].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(2 * min((means <= 0).mean(), (means >= 0).mean()))}
