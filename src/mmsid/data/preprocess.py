"""Turn the raw benchmark CSVs and metadata into Parquet tables.

Output, under $RS_DATA/processed/<name>/:

    items.parquet                 one row per item, item_idx starts at 1 (0 is padding)
    users.parquet                 user_id -> user_idx
    <split>/{train,valid,test}.parquet
                                  user_idx, history (list of item_idx, most recent last),
                                  target, timestamp
    <split>/item_pop.npy          number of training interactions per item_idx

    python -m mmsid.data.preprocess --config configs/data/video_games.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mmsid.data.download import PARTS, SPLIT_DIRS, meta_file, split_file
from mmsid.paths import processed_dir, raw_dir

# Absolute time split boundaries used by the benchmark (milliseconds).
T1 = 1628643414042
T2 = 1658002729837
MAX_HISTORY = 20


def read_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"user_id": str, "parent_asin": str, "history": str},
        keep_default_na=False,
    )
    return df[["user_id", "parent_asin", "timestamp", "history"]]


def subsample_users(users: np.ndarray, max_users: int | None, seed: int) -> np.ndarray:
    """Keep a fixed-size random subset of users. Items are never subsampled directly."""
    users = np.sort(users)
    if max_users is None or len(users) <= max_users:
        return users
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(users, size=max_users, replace=False))


def collect_items(frames: dict[tuple[str, str], pd.DataFrame]) -> list[str]:
    items: set[str] = set()
    for df in frames.values():
        items.update(df["parent_asin"].unique())
        for h in df["history"].unique():
            if h:
                items.update(h.split(" "))
    return sorted(items)


def encode_split(
    df: pd.DataFrame, item2idx: dict[str, int], user2idx: dict[str, int], max_history: int
) -> pd.DataFrame:
    """Map ids to integers and truncate histories. Rows with no history are dropped:
    there is nothing to condition on, for training or for evaluation."""
    df = df[df["history"] != ""]
    history = [[item2idx[a] for a in h.split(" ")[-max_history:]] for h in df["history"]]
    return pd.DataFrame(
        {
            "user_idx": df["user_id"].map(user2idx).to_numpy(np.int32),
            "history": history,
            "target": df["parent_asin"].map(item2idx).to_numpy(np.int32),
            "timestamp": df["timestamp"].to_numpy(np.int64),
        }
    )


def item_popularity(train: pd.DataFrame, item2idx: dict[str, int]) -> np.ndarray:
    pop = np.zeros(len(item2idx) + 1, dtype=np.int64)
    counts = train["parent_asin"].map(item2idx).value_counts()
    pop[counts.index.to_numpy()] = counts.to_numpy()
    return pop


def _price(value) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    return p if np.isfinite(p) and p > 0 else None


def _first_image(images) -> str | None:
    for img in images or []:
        for key in ("large", "hi_res", "thumb"):
            url = img.get(key)
            if url and url != "None":
                return url
    return None


def item_text(title: str, store: str, categories: list[str], price, features: list[str]) -> str:
    """One sentence per item, in the spirit of TIGER's 'title, price, brand, category'."""
    parts = [title.strip()] if title else []
    if store:
        parts.append(f"Brand: {store}")
    if categories:
        parts.append("Category: " + " > ".join(categories[-2:]))
    if price is not None:
        parts.append(f"Price: ${price:.2f}")
    if features:
        parts.append(" ".join(f.strip() for f in features[:3]))
    return ". ".join(p for p in parts if p)


def build_item_table(meta_path: Path, item2idx: dict[str, int]) -> pd.DataFrame:
    rows = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            asin = m.get("parent_asin")
            if asin not in item2idx or asin in rows:
                continue
            cats = [c for c in (m.get("categories") or []) if c]
            price = _price(m.get("price"))
            title = (m.get("title") or "").strip()
            store = (m.get("store") or "").strip()
            feats = [x for x in (m.get("features") or []) if x]
            rows[asin] = {
                "parent_asin": asin,
                "title": title,
                "store": store,
                "categories": " > ".join(cats),
                "leaf_category": cats[-1] if cats else (m.get("main_category") or ""),
                "price": price,
                "features": " ".join(feats[:3]),
                "image_url": _first_image(m.get("images")),
                "text": item_text(title, store, cats, price, feats),
            }
    empty = {
        "title": "", "store": "", "categories": "", "leaf_category": "", "price": None,
        "features": "", "image_url": None, "text": "",
    }
    table = [rows.get(a, {"parent_asin": a, **empty}) for a in item2idx]
    df = pd.DataFrame(table)
    df.insert(0, "item_idx", [item2idx[a] for a in df["parent_asin"]])
    df["has_meta"] = df["parent_asin"].isin(rows.keys())
    df["has_image"] = False  # set by mmsid.data.images once the file is on disk
    return df.sort_values("item_idx").reset_index(drop=True)


def preprocess(category: str, name: str, max_users: int | None = None, seed: int = 0) -> Path:
    src = raw_dir() / "amazon2023"
    frames = {
        (s, p): read_split_csv(src / split_file(category, s, p)) for s in SPLIT_DIRS for p in PARTS
    }

    all_users = np.unique(np.concatenate([df["user_id"].unique() for df in frames.values()]))
    users = subsample_users(all_users, max_users, seed)
    if len(users) < len(all_users):
        keep = set(users)
        frames = {k: df[df["user_id"].isin(keep)] for k, df in frames.items()}
    print(f"{category}: {len(users):,} of {len(all_users):,} users kept")

    item2idx = {a: i + 1 for i, a in enumerate(collect_items(frames))}
    user2idx = {u: i for i, u in enumerate(users)}

    out = processed_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    items = build_item_table(src / meta_file(category), item2idx)
    img_dir = raw_dir() / "images" / name
    items["has_image"] = [(img_dir / f"{i}.jpg").exists() for i in items["item_idx"]]
    items.to_parquet(out / "items.parquet", index=False)
    pd.DataFrame({"user_id": users, "user_idx": np.arange(len(users), dtype=np.int32)}).to_parquet(
        out / "users.parquet", index=False
    )

    stats = {"category": category, "users_total": int(len(all_users)), "users_kept": int(len(users)),
             "seed": seed, "items": len(item2idx), "rows": {}}
    for (split, part), df in frames.items():
        enc = encode_split(df, item2idx, user2idx, MAX_HISTORY)
        (out / split).mkdir(exist_ok=True)
        enc.to_parquet(out / split / f"{part}.parquet", index=False)
        if part == "train":
            # Training popularity counts every training interaction, including the first
            # one of each user, which has no history and so is not a training row.
            np.save(out / split / "item_pop.npy", item_popularity(df, item2idx))
        stats["rows"][f"{split}/{part}"] = {"raw": int(len(df)), "kept": int(len(enc))}
        print(f"  {split}/{part}: {len(df):,} rows, {len(enc):,} with a history")
    (out / "preprocess_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  {len(item2idx):,} items -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    preprocess(cfg.category, cfg.name, cfg.get("max_users"), cfg.get("subsample_seed", 0))


if __name__ == "__main__":
    main()
