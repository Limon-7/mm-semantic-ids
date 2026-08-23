"""Run a model over a split part and turn its top-K lists into a report."""

import time

import numpy as np
import torch

from mmsid.data.dataset import SplitPart, trim
from mmsid.eval.metrics import summarise, target_ranks
from mmsid.eval.slices import slice_metrics


@torch.no_grad()
def recommend_all(recommend, part: SplitPart, batch_size: int, device, amp: bool = True):
    """recommend(user, history) -> [B, K] item ids. Returns the stacked lists and ms per user."""
    out = []
    order = torch.argsort(part.history_len, stable=True)  # similar lengths share a batch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(0, len(part), batch_size):
        rows = order[i : i + batch_size]
        user, history = part.user[rows].to(device), trim(part.history[rows]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device != "cpu"):
            out.append(recommend(user, history).cpu().numpy())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_user = 1000 * (time.perf_counter() - start) / max(1, len(part))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # hand the beam-search memory back, the GPU is shared
    topk = np.empty_like(np.concatenate(out))
    topk[order.numpy()] = np.concatenate(out)
    return topk, ms_per_user


def diversity(topk: np.ndarray, leaf_category: np.ndarray, pop: np.ndarray, n_items: int) -> dict:
    """leaf_category: integer label per item_idx. Item 0 in a list means an empty slot."""
    real = topk > 0
    cats = leaf_category[topk]
    entropy = np.zeros(len(topk))
    for i in range(len(topk)):
        _, counts = np.unique(cats[i][real[i]], return_counts=True)
        if len(counts):
            p = counts / counts.sum()
            entropy[i] = -(p * np.log2(p)).sum()
    return {
        "coverage@10": float(len(np.unique(topk[real])) / n_items),
        "category_entropy@10": float(entropy.mean()),
        "mean_popularity@10": float(pop[topk[real]].mean()) if real.any() else 0.0,
        "empty_slots": float(1 - real.mean()),
    }


def prefix_match(ids: np.ndarray, top1: np.ndarray, target: np.ndarray) -> list[float]:
    """Share of rows whose best beam has the first 1, 2, ... tokens of the true item right."""
    same = np.cumprod(ids[top1] == ids[target], axis=1).astype(bool)
    same[top1 == 0] = False
    return [float(x) for x in same.mean(0)]


@torch.no_grad()
def invalid_beam_share(model, index, part: SplitPart, beam: int, device, n: int = 2000) -> float:
    """Share of beams that are not a real item when decoding is left unconstrained."""
    part = part.subset(torch.arange(min(n, len(part))))
    invalid, total = 0, 0
    for i in range(0, len(part), 256):
        user, history = part.user[i : i + 256].to(device), trim(part.history[i : i + 256]).to(device)
        items, _ = model.recommend(user, history, index, beam=beam, constrained=False)
        invalid += int((items == 0).sum())
        total += items.numel()
    return invalid / total


def report(topk: np.ndarray, part: SplitPart, pop: np.ndarray, has_image: np.ndarray,
           leaf_category: np.ndarray, n_items: int, ks=(5, 10)) -> tuple[dict, np.ndarray]:
    target = part.target.numpy()
    ranks = target_ranks(topk, target)
    out = summarise(ranks, ks)
    out["slices"] = slice_metrics(ranks, target, part.history_len.numpy(), pop, has_image, ks)
    out["diversity"] = diversity(topk, leaf_category, pop, n_items)
    return out, ranks
