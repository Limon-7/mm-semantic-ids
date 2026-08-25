"""Draw product images that share the first two tokens of their semantic ID.

One row per (c1, c2) prefix, eight products per row. Saved to assets/prefix_grid.png.

    uv run python scripts/prefix_grid.py data=video_games ids.spec.variant=LF
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from mmsid.config import load_config
from mmsid.data.images import image_dir
from mmsid.paths import processed_dir
from mmsid.tokenizer.fusion import IdSpec
from mmsid.train_tokenizer import ids_path

ROWS, COLS = 6, 8


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("out=")]
    out = next((a[4:] for a in sys.argv[1:] if a.startswith("out=")), "assets/prefix_grid.png")
    cfg = load_config(None, args)
    spec = IdSpec(**OmegaConf.to_container(cfg.ids.spec))
    ids = np.load(ids_path(cfg.data, cfg.split, spec))
    items = pd.read_parquet(processed_dir(cfg.data) / "items.parquet").sort_values("item_idx")
    has_image = np.concatenate([[False], items["has_image"].to_numpy()])

    df = pd.DataFrame({"item": np.arange(len(ids)), "c1": ids[:, 0], "c2": ids[:, 1]})[1:]
    df = df[has_image[df["item"]]]
    groups = [g for _, g in df.groupby(["c1", "c2"]) if len(g) >= COLS]
    rng = np.random.default_rng(0)
    chosen = [groups[i] for i in rng.choice(len(groups), ROWS, replace=False)]

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 1.6, ROWS * 1.75))
    for r, group in enumerate(chosen):
        sample = group.sample(COLS, random_state=0)
        for c, item in enumerate(sample["item"]):
            ax = axes[r, c]
            ax.imshow(Image.open(image_dir(cfg.data) / f"{item}.jpg"))
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"({group['c1'].iloc[0]}, {group['c2'].iloc[0]}, ·, ·)", fontsize=9)
    fig.suptitle(f"Products that share the first two tokens of their ID  ({cfg.data}, {spec.name})",
                 fontsize=11)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
