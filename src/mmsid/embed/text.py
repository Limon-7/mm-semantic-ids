"""Text embeddings for every item, aligned to item_idx (row 0 is the padding row).

    python -m mmsid.embed.text --name video_games --config configs/embed/sent_t5.yaml
"""

import argparse

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

from mmsid.paths import features_dir, processed_dir


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def embed_texts(texts: list[str], model_name: str, batch_size: int, max_tokens: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_tokens
    emb = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def with_padding_row(emb: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1, emb.shape[1]), emb.dtype), emb]).astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    items = pd.read_parquet(processed_dir(args.name) / "items.parquet").sort_values("item_idx")
    assert (items["item_idx"].to_numpy() == np.arange(1, len(items) + 1)).all()
    texts = [t if t else "unknown product" for t in items["text"]]
    emb = embed_texts(texts, cfg.model, cfg.batch_size, cfg.max_tokens)

    out = features_dir(args.name)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"text_{cfg.tag}.npy", with_padding_row(emb))
    print(f"{args.name}: text_{cfg.tag}.npy {emb.shape}")


if __name__ == "__main__":
    main()
