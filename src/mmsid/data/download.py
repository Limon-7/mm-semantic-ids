"""Fetch the Amazon Reviews 2023 benchmark splits and raw item metadata for one category.

The files come straight from the official Hugging Face dataset repo. Nothing is rehosted.

    python -m mmsid.data.download --category Video_Games
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from mmsid.paths import raw_dir

REPO = "McAuley-Lab/Amazon-Reviews-2023"
SPLIT_DIRS = {"last_out": "last_out_w_his", "timestamp": "timestamp_w_his"}
PARTS = ("train", "valid", "test")


def split_file(category: str, split: str, part: str) -> str:
    return f"benchmark/5core/{SPLIT_DIRS[split]}/{category}.{part}.csv"


def meta_file(category: str) -> str:
    return f"raw/meta_categories/meta_{category}.jsonl"


def download_category(category: str, with_meta: bool = True) -> list[Path]:
    out = raw_dir() / "amazon2023"
    out.mkdir(parents=True, exist_ok=True)
    names = [split_file(category, s, p) for s in SPLIT_DIRS for p in PARTS]
    if with_meta:
        names.append(meta_file(category))
    paths = []
    for name in names:
        path = hf_hub_download(REPO, repo_type="dataset", filename=name, local_dir=out)
        print(f"ok  {name}")
        paths.append(Path(path))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--no-meta", action="store_true")
    args = ap.parse_args()
    download_category(args.category, with_meta=not args.no_meta)


if __name__ == "__main__":
    main()
