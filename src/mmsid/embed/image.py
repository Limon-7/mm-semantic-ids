"""SigLIP image embeddings for every item, aligned to item_idx (row 0 is the padding row).

Items without a downloaded image get the embedding of their title from the same model's text
tower, which lives in the same space. `has_image` in items.parquet says which is which.

    python -m mmsid.embed.image --name video_games --config configs/embed/siglip.yaml
"""

import argparse

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from mmsid.data.images import image_dir
from mmsid.embed.text import l2_normalise, with_padding_row
from mmsid.paths import features_dir, processed_dir


class ImageFiles(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            ok = True
        except Exception:
            img, ok = Image.new("RGB", (224, 224)), False
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0], ok


def _features(out) -> torch.Tensor:
    return out if torch.is_tensor(out) else out.pooler_output


@torch.no_grad()
def embed_images(paths, model, processor, batch_size: int, workers: int, device: str):
    loader = DataLoader(ImageFiles(paths, processor), batch_size=batch_size, num_workers=workers)
    feats, oks = [], []
    for pixels, ok in tqdm(loader, desc="image tower", mininterval=5):
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            f = _features(model.get_image_features(pixel_values=pixels.to(device)))
        feats.append(f.float().cpu().numpy())
        oks.append(ok.numpy())
    return np.concatenate(feats), np.concatenate(oks)


@torch.no_grad()
def embed_titles(titles, model, processor, batch_size: int, device: str) -> np.ndarray:
    feats = []
    for i in range(0, len(titles), batch_size):
        tok = processor(text=titles[i : i + batch_size], padding="max_length", truncation=True,
                        max_length=64, return_tensors="pt").to(device)
        feats.append(_features(model.get_text_features(input_ids=tok["input_ids"])).float().cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items_path = processed_dir(args.name) / "items.parquet"
    items = pd.read_parquet(items_path).sort_values("item_idx").reset_index(drop=True)
    model = AutoModel.from_pretrained(cfg.model).to(device).eval()
    processor = AutoProcessor.from_pretrained(cfg.model)

    dim = model.config.vision_config.hidden_size
    emb = np.zeros((len(items), dim), np.float32)
    has = items["has_image"].to_numpy().copy()

    rows = np.flatnonzero(has)
    if len(rows):
        paths = [image_dir(args.name) / f"{i}.jpg" for i in items["item_idx"].to_numpy()[rows]]
        feats, ok = embed_images(paths, model, processor, cfg.batch_size, cfg.workers, device)
        emb[rows[ok]] = feats[ok]
        has[rows[~ok]] = False  # unreadable file: treat as no image

    rows = np.flatnonzero(~has)
    if len(rows):
        titles = [t if t else "unknown product" for t in items["title"].to_numpy()[rows]]
        emb[rows] = embed_titles(titles, model, processor, cfg.batch_size, device)

    items["has_image"] = has
    items.to_parquet(items_path, index=False)
    out = features_dir(args.name)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"image_{cfg.tag}.npy", with_padding_row(l2_normalise(emb)))
    print(f"{args.name}: image_{cfg.tag}.npy {emb.shape}, real images for {has.mean():.1%} of items")


if __name__ == "__main__":
    main()
