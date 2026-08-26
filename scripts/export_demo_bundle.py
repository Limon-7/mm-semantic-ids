"""Pack what the demo needs for one category into a single folder.

    uv run python scripts/export_demo_bundle.py --data video_games --tag vg_lastout --out demo/bundle

The bundle holds the item table (titles and image URLs, no image files), the semantic IDs, the
trained checkpoints and quantisers for the text-only and late-fusion variants, and the item
embeddings used for nearest-neighbour lookups. It is not committed.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mmsid.paths import checkpoints_dir, features_dir, processed_dir
from mmsid.tokenizer.assign import level_sizes
from mmsid.tokenizer.fusion import IdSpec
from mmsid.train_tokenizer import ids_path

VARIANTS = ("T", "LF")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="video_games")
    ap.add_argument("--tag", default="vg_lastout", help="prefix of the experiment names")
    ap.add_argument("--out", default="demo/bundle")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(processed_dir(args.data) / "items.parquet").sort_values("item_idx")
    items[["item_idx", "title", "store", "leaf_category", "image_url"]].to_parquet(
        out / "items.parquet", index=False)

    meta = {"data": args.data, "variants": {}}
    for variant in VARIANTS:
        name = f"{args.tag}_{variant}"
        cfg = OmegaConf.load(Path("results") / name / "seed0" / "config.yaml")
        spec = IdSpec(**OmegaConf.to_container(cfg.ids.spec))
        src = ids_path(cfg.data, cfg.split, spec)
        ids = np.load(src)
        np.save(out / f"ids_{variant}.npy", ids)
        shutil.copy(checkpoints_dir() / name / "seed0" / "best.pt", out / f"tiger_{variant}.pt")

        with open(src.with_suffix(".quantizers.pkl"), "rb") as f:
            fitted = pickle.load(f)
        for _, q in fitted:  # make the quantisers loadable on a machine without a GPU
            if hasattr(q, "model"):
                q.model.cpu()
                q.device = "cpu"
        with open(out / f"quantizers_{variant}.pkl", "wb") as f:
            pickle.dump(fitted, f)

        meta["variants"][variant] = {
            "experiment": name, "id": spec.name,
            "level_sizes": level_sizes(ids, [spec.codebook_size] * spec.levels),
            "tiger": OmegaConf.to_container(cfg.model.tiger),
        }
        meta["text"], meta["image"] = cfg.ids.text, cfg.ids.image

    for kind, tag in (("text", meta["text"]), ("image", meta["image"])):
        shutil.copy(features_dir(args.data) / f"{kind}_{tag}.npy", out / f"{kind}.npy")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in out.iterdir()) / 2**20
    print(f"bundle written to {out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
