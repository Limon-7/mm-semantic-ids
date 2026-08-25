"""Paired bootstrap over test users for the comparisons the project is about.

For every dataset and split: each multimodal ID variant against text-only IDs, and the best
generative model against the best SASRec (both chosen on validation NDCG@10). Reads the per-row
ranks that train_seq saves next to the checkpoints and writes results/analysis/bootstrap.json.

    uv run python scripts/analyse.py
"""

import json
from pathlib import Path

import numpy as np

from mmsid.eval.metrics import ndcg_at_k, paired_bootstrap
from mmsid.paths import checkpoints_dir

RESULTS = Path("results")
GENERATIVE = ("T", "I", "EF", "LF")
SASREC = ("sasrec_id", "sasrec_text", "sasrec_text_image", "sasrec_id_text_image")
SEED = 0


def metrics(name: str) -> dict | None:
    path = RESULTS / name / f"seed{SEED}" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def ndcg_rows(name: str) -> np.ndarray:
    return ndcg_at_k(np.load(checkpoints_dir() / name / f"seed{SEED}" / "test_ranks.npy"), 10)


def best_on_validation(names: list[str]) -> str | None:
    scored = [(m["valid"]["ndcg@10"], n) for n in names if (m := metrics(n))]
    return max(scored)[1] if scored else None


def main() -> None:
    rows = []
    for data in ("vg", "toys"):
        for split in ("lastout", "time"):
            tag = f"{data}_{split}"
            pairs = [(f"{tag}_{v}", f"{tag}_T") for v in ("LF", "EF", "I")]
            gen = best_on_validation([f"{tag}_{v}" for v in GENERATIVE])
            dense = best_on_validation([f"{tag}_{v}" for v in SASREC])
            if gen and dense:
                pairs.append((gen, dense))
            for a, b in pairs:
                if metrics(a) and metrics(b):
                    out = paired_bootstrap(ndcg_rows(a), ndcg_rows(b))
                    rows.append({"data": data, "split": split, "a": a, "b": b, **out})
                    print(f"{a:28s} vs {b:28s} {out['mean_diff']:+.4f} "
                          f"[{out['ci_low']:+.4f}, {out['ci_high']:+.4f}] p={out['p_two_sided']:.3f}")
    out_dir = RESULTS / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bootstrap.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
