"""Amazon Beauty 2014 5-core, the dataset the TIGER paper reports on.

Used only for the sanity reproduction. The files are fetched from the original SNAP host and
written in the same Parquet layout as the 2023 data, with the standard leave-one-out protocol:
the last item of each user is the test target, the second-last is validation, and every
earlier position is a training target.

    python -m mmsid.data.beauty2014
"""

import ast
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from mmsid.data.preprocess import MAX_HISTORY, _price, item_text
from mmsid.paths import processed_dir, raw_dir

BASE = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
REVIEWS = "reviews_Beauty_5.json.gz"
META = "meta_Beauty.json.gz"
NAME = "beauty2014"


def fetch(filename: str) -> Path:
    dest = raw_dir() / "amazon2014" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {filename}")
        urllib.request.urlretrieve(BASE + filename, dest)
    return dest


def load_reviews(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
    return pd.DataFrame(rows, columns=["user_id", "asin", "timestamp"])


def leave_one_out(sequences: dict[int, list[int]], times: dict[int, list[int]], max_history: int):
    """Split each user sequence into train prefixes, one validation row and one test row."""
    parts = {"train": [], "valid": [], "test": []}
    for u, seq in sequences.items():
        n = len(seq)
        if n < 3:
            continue
        for pos in range(1, n):
            part = "test" if pos == n - 1 else "valid" if pos == n - 2 else "train"
            parts[part].append((u, seq[max(0, pos - max_history):pos], seq[pos], times[u][pos]))
    cols = ["user_idx", "history", "target", "timestamp"]
    return {p: pd.DataFrame(rows, columns=cols) for p, rows in parts.items()}


def build_items(meta_path: Path, item2idx: dict[str, int]) -> pd.DataFrame:
    rows = {}
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            # the 2014 metadata is written as Python dict literals, not JSON
            asin_pos = line.find("'asin': '")
            asin = line[asin_pos + 9 : asin_pos + 19] if asin_pos >= 0 else None
            if asin not in item2idx:
                continue
            m = ast.literal_eval(line)
            cats = (m.get("categories") or [[]])[0]
            price = _price(m.get("price"))
            title = (m.get("title") or "").strip()
            brand = (m.get("brand") or "").strip()
            rows[m["asin"]] = {
                "parent_asin": m["asin"], "title": title, "store": brand,
                "categories": " > ".join(cats), "leaf_category": cats[-1] if cats else "",
                "price": price, "features": "", "image_url": m.get("imUrl"),
                "text": item_text(title, brand, cats, price, []),
            }
    empty = {"title": "", "store": "", "categories": "", "leaf_category": "", "price": None,
             "features": "", "image_url": None, "text": ""}
    df = pd.DataFrame([rows.get(a, {"parent_asin": a, **empty}) for a in item2idx])
    df.insert(0, "item_idx", [item2idx[a] for a in df["parent_asin"]])
    df["has_meta"] = df["parent_asin"].isin(rows.keys())
    df["has_image"] = False
    return df


def main() -> None:
    reviews = load_reviews(fetch(REVIEWS))
    reviews = reviews.sort_values(["user_id", "timestamp"], kind="stable")
    item2idx = {a: i + 1 for i, a in enumerate(sorted(reviews["asin"].unique()))}
    users = np.sort(reviews["user_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    reviews["item_idx"] = reviews["asin"].map(item2idx)
    reviews["user_idx"] = reviews["user_id"].map(user2idx)

    grouped = reviews.groupby("user_idx", sort=True)
    sequences = grouped["item_idx"].apply(list).to_dict()
    times = grouped["timestamp"].apply(list).to_dict()
    parts = leave_one_out(sequences, times, MAX_HISTORY)

    out = processed_dir(NAME)
    (out / "last_out").mkdir(parents=True, exist_ok=True)
    for part, df in parts.items():
        df.to_parquet(out / "last_out" / f"{part}.parquet", index=False)

    # training popularity: everything except each user's last two items
    pop = np.zeros(len(item2idx) + 1, dtype=np.int64)
    for seq in sequences.values():
        for i in seq[:-2]:
            pop[i] += 1
    np.save(out / "last_out" / "item_pop.npy", pop)

    build_items(fetch(META), item2idx).to_parquet(out / "items.parquet", index=False)
    pd.DataFrame({"user_id": users, "user_idx": np.arange(len(users), dtype=np.int32)}).to_parquet(
        out / "users.parquet", index=False
    )
    print(f"{NAME}: {len(users):,} users, {len(item2idx):,} items, {len(reviews):,} interactions")
    print({p: len(df) for p, df in parts.items()})


if __name__ == "__main__":
    main()
