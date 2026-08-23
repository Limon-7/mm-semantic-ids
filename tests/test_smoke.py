"""A tiny end-to-end run on synthetic data: tokeniser, training, evaluation, metrics.json."""

import json

import numpy as np
import pandas as pd
import pytest

from mmsid.config import load_config
from mmsid.train_seq import run

N_ITEMS, N_USERS, DIM = 120, 250, 16


def _write_dataset(root):
    rng = np.random.default_rng(0)
    proc = root / "processed" / "toy"
    feat = root / "features" / "toy"
    (proc / "last_out").mkdir(parents=True)
    feat.mkdir(parents=True)

    pd.DataFrame({
        "item_idx": np.arange(1, N_ITEMS + 1),
        "leaf_category": [f"cat{i % 6}" for i in range(N_ITEMS)],
        "has_image": rng.random(N_ITEMS) < 0.8,
    }).to_parquet(proc / "items.parquet", index=False)

    # each user walks through the catalogue in steps of 1, so the next item is learnable
    rows = {"train": [], "valid": [], "test": []}
    pop = np.zeros(N_ITEMS + 1, np.int64)
    for u in range(N_USERS):
        start = int(rng.integers(1, N_ITEMS + 1))
        seq = [(start + k - 1) % N_ITEMS + 1 for k in range(7)]
        for pos in range(1, 7):
            part = "test" if pos == 6 else "valid" if pos == 5 else "train"
            rows[part].append((u, seq[:pos], seq[pos], pos))
        for i in seq[:5]:
            pop[i] += 1
    for part, data in rows.items():
        pd.DataFrame(data, columns=["user_idx", "history", "target", "timestamp"]).to_parquet(
            proc / "last_out" / f"{part}.parquet", index=False)
    np.save(proc / "last_out" / "item_pop.npy", pop)

    for name in ("text_st5", "image_siglip"):
        x = rng.normal(size=(N_ITEMS + 1, DIM)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x[0] = 0
        np.save(feat / f"{name}.npy", x.astype(np.float16))


@pytest.fixture()
def toy(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setenv("RS_DATA", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


COMMON = ["data=toy", "split=last_out", "train.steps=150", "train.eval_every=50", "train.warmup=10",
          "train.batch_size=64", "train.eval_users=100", "eval.beam=10", "eval.batch_size=64"]


def test_generative_model_end_to_end(toy):
    cfg = load_config(None, COMMON + [
        "name=toy_tiger", "model.type=tiger", "ids.spec.variant=LF", "ids.spec.quantizer=rqkmeans",
        "ids.spec.codebook_size=8", "model.tiger.d_model=32", "model.tiger.d_ff=64",
        "model.tiger.d_kv=16", "model.tiger.heads=2", "model.tiger.layers=1",
        "model.tiger.user_tokens=16"])
    metrics = run(cfg)
    saved = json.loads((toy / "results/toy_tiger/seed0/metrics.json").read_text())
    assert saved["test"]["n"] == N_USERS and saved["config_hash"] == metrics["config_hash"]
    assert 0 <= saved["test"]["recall@10"] <= 1
    assert len(saved["test"]["prefix_match"]) == 5          # 4 content tokens + collision token
    assert set(saved["test"]["slices"]) >= {"pop/cold", "pop/tail", "history/under_5", "image/real"}
    assert (toy / "results/tokenizer/toy/last_out/LF_rqkmeans_4x8_t2i2.json").exists()
    assert run(cfg) is None                                 # a finished run is skipped


def test_sasrec_learns_the_synthetic_pattern(toy):
    cfg = load_config(None, COMMON + [
        "name=toy_sasrec", "model.type=sasrec", "model.sasrec.dim=32", "model.sasrec.layers=1",
        "model.sasrec.dropout=0.0", "train.steps=800", "train.eval_every=200", "train.lr=0.01",
        "train.weight_decay=0"])
    metrics = run(cfg)
    assert metrics["test"]["recall@10"] > 0.5               # next item is always "current + 1"


def test_interrupted_run_continues_from_its_last_validation_step(toy, monkeypatch, capsys):
    import mmsid.train_seq as train_seq

    cfg = load_config(None, COMMON + [
        "name=toy_resume", "model.type=sasrec", "model.sasrec.dim=32", "model.sasrec.layers=1"])
    real, calls = train_seq.target_ranks, []

    def crash_on_second_validation(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(train_seq, "target_ranks", crash_on_second_validation)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)                                            # dies at step 100 of 150
    out = toy / "results/toy_resume/seed0"
    assert not (out / "metrics.json").exists() and not (out / "RUNNING").exists()

    monkeypatch.setattr(train_seq, "target_ranks", real)
    metrics = run(cfg)
    assert "resumed toy_resume from step 50" in capsys.readouterr().out
    assert [row["step"] for row in metrics["history"]] == [50, 100, 150]
    assert metrics["steps_run"] == 150
    assert not (toy / "checkpoints/toy_resume/seed0/last.pt").exists()


def test_popularity_baseline(toy):
    metrics = run(load_config(None, COMMON + ["name=toy_pop", "model.type=popularity"]))
    assert set(metrics) >= {"popularity", "popularity_recent"}
    assert metrics["popularity"]["test"]["n"] == N_USERS
