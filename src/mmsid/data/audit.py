"""Write a Markdown audit of one processed dataset to docs/audit/<name>.md.

    python -m mmsid.data.audit --name video_games
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmsid.paths import processed_dir

SPLITS = ("last_out", "timestamp")


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def split_stats(root: Path, split: str) -> dict:
    train = pd.read_parquet(root / split / "train.parquet")
    test = pd.read_parquet(root / split / "test.parquet")
    valid = pd.read_parquet(root / split / "valid.parquet")
    pop = np.load(root / split / "item_pop.npy")
    repeat = np.mean([t in set(h) for t, h in zip(test["target"], test["history"], strict=True)])
    return {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "train_items": int((pop > 0).sum()),
        "cold_test_share": float((pop[test["target"].to_numpy()] == 0).mean()),
        "cold_valid_share": float((pop[valid["target"].to_numpy()] == 0).mean()),
        "repeat_test_share": float(repeat),
        "pop": pop,
    }


def audit(name: str) -> str:
    root = processed_dir(name)
    items = pd.read_parquet(root / "items.parquet")
    users = pd.read_parquet(root / "users.parquet")
    pre = json.loads((root / "preprocess_stats.json").read_text())
    stats = {s: split_stats(root, s) for s in SPLITS}

    n_inter = sum(pre["rows"][f"last_out/{p}"]["raw"] for p in ("train", "valid", "test"))
    lo = pd.concat(
        [pd.read_parquet(root / "last_out" / f"{p}.parquet", columns=["user_idx"])
         for p in ("train", "valid", "test")]
    )
    seq_len = lo["user_idx"].value_counts().to_numpy() + 1  # + the first, history-less interaction
    bins = [(5, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]

    pop = stats["last_out"]["pop"][1:]
    order = np.sort(pop)[::-1]
    top1 = order[: max(1, len(order) // 100)].sum() / max(1, order.sum())
    top10 = order[: max(1, len(order) // 10)].sum() / max(1, order.sum())

    L = [f"# Data audit: `{name}`", ""]
    L += [f"Source category: `{pre['category']}`, 5-core benchmark files."]
    if pre["users_kept"] < pre["users_total"]:
        L += [f"Users subsampled: {pre['users_kept']:,} of {pre['users_total']:,} (seed {pre['seed']})."]
    L += ["", "| | |", "|---|---|",
          f"| Users | {len(users):,} |",
          f"| Items | {len(items):,} |",
          f"| Interactions | {n_inter:,} |",
          f"| Density | {n_inter / (len(users) * len(items)):.5%} |",
          f"| Items found in metadata | {items['has_meta'].mean():.1%} |",
          f"| Items with an image URL | {items['image_url'].notna().mean():.1%} |",
          f"| Items with a downloaded image | {items['has_image'].mean():.1%} |",
          f"| Items with a non-empty title | {(items['title'] != '').mean():.1%} |",
          f"| Distinct leaf categories | {items['leaf_category'].nunique():,} |", ""]

    L += ["## Sequence length (interactions per user)", "", "| Length | Users | Share |", "|---|---|---|"]
    for a, b in bins:
        n = int(((seq_len >= a) & (seq_len <= b)).sum())
        label = f"{a}" if a == b else (f"{a}+" if b > 10**6 else f"{a}-{b}")
        L += [f"| {label} | {n:,} | {n / len(seq_len):.1%} |"]
    L += ["", f"Mean {seq_len.mean():.1f}, median {np.median(seq_len):.0f}, max {seq_len.max()}.", ""]

    L += ["## Item popularity (training interactions, leave-last-out)", "",
          f"- Median {np.median(pop):.0f}, mean {pop.mean():.1f}, max {pop.max():,}.",
          f"- Top 1% of items hold {top1:.1%} of training interactions, top 10% hold {top10:.1%}.",
          f"- Gini coefficient {gini(pop):.3f}.", ""]

    L += ["## Splits", "",
          "| Split | Train rows | Valid rows | Test rows | Items seen in training | Cold share (valid) | Cold share (test) | Target already in history (test) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        st = stats[s]
        L += [f"| `{s}` | {st['train_rows']:,} | {st['valid_rows']:,} | {st['test_rows']:,} | "
              f"{st['train_items']:,} | {st['cold_valid_share']:.1%} | {st['cold_test_share']:.1%} | "
              f"{st['repeat_test_share']:.2%} |"]
    L += ["", "Rows are `(history, target)` pairs with a non-empty history. "
          "A cold target is an item with no interaction in that split's training file.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="docs/audit")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = audit(args.name)
    (out / f"{args.name}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
