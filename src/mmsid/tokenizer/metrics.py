"""Quality measures for a set of semantic IDs."""

import numpy as np
import pandas as pd


def codebook_usage(codes: np.ndarray, size: int) -> float:
    return len(np.unique(codes)) / size


def perplexity(codes: np.ndarray) -> float:
    p = np.bincount(codes) / len(codes)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def collision_stats(content_codes: np.ndarray) -> dict:
    df = pd.DataFrame(content_codes)
    sizes = df.groupby(list(df.columns), sort=False).size().to_numpy()
    return {
        "collision_rate": float((sizes - 1).sum() / len(content_codes)),
        "items_in_shared_code": float(sizes[sizes > 1].sum() / len(content_codes)),
        "largest_group": int(sizes.max()),
        "distinct_codes": int(len(sizes)),
    }


def pair_agreement(groups: np.ndarray, labels: np.ndarray) -> float:
    """Probability that two different items drawn from the same group share a label."""
    df = pd.DataFrame({"g": groups, "c": labels})
    n_gc = df.groupby(["g", "c"]).size()
    n_g = df.groupby("g").size()
    same = (n_gc * (n_gc - 1)).sum()
    total = (n_g * (n_g - 1)).sum()
    return float(same / total) if total else float("nan")


def prefix_purity(content_codes: np.ndarray, labels: np.ndarray, depth: int = 2) -> dict:
    """How often two items with the same first `depth` tokens share a leaf category,
    next to the same number for two random items (the chance level)."""
    keep = labels != ""
    codes, labels = content_codes[keep], labels[keep]
    prefix = pd.MultiIndex.from_arrays(codes[:, :depth].T).factorize()[0]
    return {
        "prefix_purity": pair_agreement(prefix, labels),
        "chance_purity": pair_agreement(np.zeros(len(labels), int), labels),
    }


def tokenizer_report(content_codes: np.ndarray, codebook_sizes: list[int], labels: np.ndarray,
                     recon_err: np.ndarray, train_mask: np.ndarray) -> dict:
    """content_codes, labels, recon_err and train_mask are aligned, one row per real item."""
    report = {
        "n_items": int(len(content_codes)),
        "n_train_items": int(train_mask.sum()),
        "usage": [codebook_usage(content_codes[:, i], k) for i, k in enumerate(codebook_sizes)],
        "perplexity": [perplexity(content_codes[:, i]) for i in range(content_codes.shape[1])],
        "recon_err_train": float(np.mean(recon_err[train_mask])),
        "recon_err_unseen": float(np.mean(recon_err[~train_mask])) if (~train_mask).any() else None,
    }
    report.update(collision_stats(content_codes))
    report.update(prefix_purity(content_codes, labels))
    return report
