"""Train and evaluate one sequence model.

    python -m mmsid.train_seq vg_lastout_LF            # configs/exp/vg_lastout_LF.yaml
    python -m mmsid.train_seq vg_lastout_LF seed=1 train.steps=2000

Writes results/<name>/seed<k>/metrics.json. A run whose metrics.json exists is skipped.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from mmsid.config import config_hash, load_config
from mmsid.data.dataset import BucketSampler, load_item_pop, load_part, trim
from mmsid.data.preprocess import MAX_HISTORY
from mmsid.eval.evaluate import invalid_beam_share, prefix_match, recommend_all, report
from mmsid.eval.metrics import summarise, target_ranks
from mmsid.models.popularity import overall_top, recent_top
from mmsid.models.sasrec import SASRec
from mmsid.models.tiger import Tiger
from mmsid.models.trie import PrefixIndex
from mmsid.paths import checkpoints_dir, features_dir, processed_dir
from mmsid.tokenizer.assign import level_sizes
from mmsid.tokenizer.fusion import IdSpec
from mmsid.train_tokenizer import build_ids

TOP_K = 10


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def sasrec_content(data: str, cfg) -> torch.Tensor | None:
    kind = cfg.model.sasrec.item_input
    if kind == "id":
        return None
    feats = [np.load(features_dir(data) / f"text_{cfg.ids.text}.npy").astype(np.float32)]
    if "image" in kind:
        feats.append(np.load(features_dir(data) / f"image_{cfg.ids.image}.npy").astype(np.float32))
    return torch.from_numpy(np.concatenate(feats, axis=1))


def run(cfg) -> dict | None:
    out_dir = Path("results") / cfg.name / f"seed{cfg.seed}"
    if (out_dir / "metrics.json").exists():
        print(f"skip {out_dir}: metrics.json exists")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = checkpoints_dir() / cfg.name / f"seed{cfg.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    start = time.time()

    data, split, ks = cfg.data, cfg.split, tuple(cfg.eval.ks)
    items = pd.read_parquet(processed_dir(data) / "items.parquet").sort_values("item_idx")
    N = len(items)
    pop = load_item_pop(data, split)
    has_image = np.concatenate([[False], items["has_image"].to_numpy()])
    leaf = np.concatenate([[0], pd.factorize(items["leaf_category"])[0] + 1])
    train, valid, test = (load_part(data, split, p) for p in ("train", "valid", "test"))

    def final_report(recommend, part, amp=True):
        topk, ms = recommend_all(recommend, part, cfg.eval.batch_size, device, amp)
        rep, ranks = report(topk, part, pop, has_image, leaf, N, ks)
        rep["ms_per_user"] = ms
        return rep, ranks, topk

    metrics = {"name": cfg.name, "data": data, "split": split, "seed": cfg.seed,
               "model": cfg.model.type, "config_hash": config_hash(cfg), "git_commit": git_commit()}

    if cfg.model.type == "popularity":
        lists = {"popularity": overall_top(pop, TOP_K),
                 "popularity_recent": recent_top(data, split, N, TOP_K)}
        for key, top in lists.items():
            def rec(user, history, top=top):
                return torch.from_numpy(top).expand(len(user), -1)
            metrics[key] = {"valid": final_report(rec, valid, False)[0],
                            "test": final_report(rec, test, False)[0]}
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    if cfg.model.type == "tiger":
        ids = np.load(build_ids(cfg))
        spec = IdSpec(**OmegaConf.to_container(cfg.ids.spec))
        sizes = level_sizes(ids, [spec.codebook_size] * spec.levels)
        model = Tiger(ids, sizes, **cfg.model.tiger).to(device)
        index = PrefixIndex(ids, sizes, device)
        metrics["ids"] = {"name": spec.name, "level_sizes": sizes}

        def recommender(beam):
            return lambda u, h: model.recommend(u, h, index, beam=beam)[0][:, :TOP_K]
    else:
        content = sasrec_content(data, cfg)
        model = SASRec(N, MAX_HISTORY, content=content, **cfg.model.sasrec).to(device)

        def recommender(beam):
            return lambda u, h: model.recommend(u, h, TOP_K)[0]

    tr = cfg.train
    opt = torch.optim.AdamW(model.parameters(), lr=tr.lr, weight_decay=tr.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, tr.warmup, tr.steps))
    writer = SummaryWriter(ckpt_dir / "tb")

    train = train.to(device)
    gen = torch.Generator().manual_seed(0)  # the validation sample is the same for every run
    val_sample = valid.subset(torch.randperm(len(valid), generator=gen)[: tr.eval_users])
    sampler = BucketSampler(train.history_len, seed=cfg.seed)

    best, best_step, bad, history, loss_sum = -1.0, 0, 0, [], 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(1, tr.steps + 1):
        idx = sampler.sample(tr.batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tr.amp and device == "cuda"):
            loss = model(train.user[idx], trim(train.history[idx]), train.target[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_sum += loss.item()

        if step % tr.eval_every == 0 or step == tr.steps:
            model.eval()
            topk, _ = recommend_all(recommender(tr.eval_beam), val_sample, cfg.eval.batch_size,
                                    device, tr.amp)
            val = summarise(target_ranks(topk, val_sample.target.numpy()), ks)
            model.train()
            n = tr.eval_every if step % tr.eval_every == 0 else step % tr.eval_every
            row = {"step": step, "loss": loss_sum / n, "val_ndcg@10": val["ndcg@10"],
                   "val_recall@10": val["recall@10"], "minutes": (time.time() - start) / 60}
            loss_sum = 0.0
            history.append(row)
            for key, value in row.items():
                writer.add_scalar(key, value, step)
            print(json.dumps(row), flush=True)
            if val["ndcg@10"] > best:
                best, best_step, bad = val["ndcg@10"], step, 0
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
            else:
                bad += 1
                if bad >= tr.patience:
                    break
    writer.close()
    train_minutes = (time.time() - start) / 60

    model.load_state_dict(torch.load(ckpt_dir / "best.pt"))
    model.eval()
    val_rep, _, _ = final_report(recommender(cfg.eval.beam), valid, tr.amp)
    run_info = {
        "best_step": best_step, "steps_run": step, "best_val_sample_ndcg@10": best,
        "train_minutes": train_minutes, "train_gpu_hours": train_minutes / 60,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0,
        "parameters": sum(p.numel() for p in model.parameters()), "history": history,
    }
    if not cfg.eval.test:
        metrics.update({"valid": val_rep, **run_info})
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({k: v for k, v in val_rep.items() if "@" in k}))
        return metrics
    test_rep, ranks, topk = final_report(recommender(cfg.eval.beam), test, tr.amp)
    np.save(ckpt_dir / "test_ranks.npy", ranks)
    np.save(ckpt_dir / "test_topk.npy", topk)

    if cfg.model.type == "tiger":
        test_rep["prefix_match"] = prefix_match(ids, topk[:, 0], test.target.numpy())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tr.amp and device == "cuda"):
            test_rep["invalid_beam_share_unconstrained"] = invalid_beam_share(
                model, index, test, cfg.eval.beam, device)

    metrics.update({"valid": val_rep, "test": test_rep, **run_info,
                    "wall_clock_minutes": (time.time() - start) / 60})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: test_rep[k] for k in test_rep if "@" in k}))
    return metrics


def main() -> None:
    exp = sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else None
    overrides = sys.argv[2:] if exp else sys.argv[1:]
    run(load_config(exp, overrides))


if __name__ == "__main__":
    main()
