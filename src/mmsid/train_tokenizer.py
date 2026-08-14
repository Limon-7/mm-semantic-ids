"""Build semantic IDs for one dataset, split and ID variant, and write the tokeniser report.

    python -m mmsid.train_tokenizer data=video_games split=last_out ids.variant=LF

Writes $RS_DATA/features/<data>/ids/<split>/<id name>.npy  ([n_items + 1, levels + 1], row 0 is
padding) and results/tokenizer/<data>/<split>/<id name>.json.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mmsid.config import load_config
from mmsid.paths import features_dir, processed_dir
from mmsid.tokenizer.assign import add_collision_token, level_sizes, with_padding_row
from mmsid.tokenizer.fusion import IdSpec, build_content_codes
from mmsid.tokenizer.metrics import tokenizer_report


def ids_path(data: str, split: str, spec: IdSpec, seed: int = 0) -> Path:
    suffix = "" if seed == 0 else f"_s{seed}"
    return features_dir(data) / "ids" / split / f"{spec.name}{suffix}.npy"


def load_features(data: str, tag: str, kind: str) -> np.ndarray:
    return np.load(features_dir(data) / f"{kind}_{tag}.npy").astype(np.float32)[1:]


def build_ids(cfg, results_root: str = "results") -> Path:
    spec = IdSpec(**OmegaConf.to_container(cfg.ids.spec))
    data, split, seed = cfg.data, cfg.split, cfg.get("id_seed", 0)
    out = ids_path(data, split, spec, seed)
    if out.exists():
        return out

    items = pd.read_parquet(processed_dir(data) / "items.parquet").sort_values("item_idx")
    train_mask = np.load(processed_dir(data) / split / "item_pop.npy")[1:] > 0
    text = load_features(data, cfg.ids.text, "text") if spec.variant != "I" else None
    image = load_features(data, cfg.ids.image, "image") if spec.variant != "T" else None

    start = time.time()
    qcfg = OmegaConf.to_container(cfg.ids.get(spec.quantizer, {}))
    codes, err, fitted = build_content_codes(spec, text, image, train_mask, seed, qcfg)
    ids = add_collision_token(codes)
    sizes = level_sizes(ids, [spec.codebook_size] * spec.levels)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, with_padding_row(ids))
    with open(out.with_suffix(".quantizers.pkl"), "wb") as f:
        pickle.dump(fitted, f)

    report = tokenizer_report(codes, sizes[:-1], items["leaf_category"].to_numpy(), err, train_mask)
    report.update({"data": data, "split": split, "id": spec.name, "seed": seed,
                   "level_sizes": sizes, "seconds": round(time.time() - start, 1),
                   "text": cfg.ids.text, "image": cfg.ids.image})
    for _, q in fitted:
        if getattr(q, "log", None):
            report.setdefault("train_log_last", []).append(q.log[-1])
    rep_path = Path(results_root) / "tokenizer" / data / split / f"{out.stem}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return out


def main() -> None:
    build_ids(load_config(None, sys.argv[1:]))


if __name__ == "__main__":
    main()
