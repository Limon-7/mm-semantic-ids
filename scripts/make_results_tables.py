"""Build every results table from results/**/metrics.json and write results/tables.md.

Numbers in the README are pasted from this file's output, never typed by hand.

    uv run python scripts/make_results_tables.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("results")
DATASETS = {"vg": "Video_Games", "toys": "Toys_and_Games"}
SPLITS = {"lastout": "leave-last-out", "time": "time split"}
VARIANTS = {"T": "Text only (TIGER)", "I": "Image only", "EF": "Text + image, early fusion",
            "LF": "Text + image, late fusion"}
SASREC = {"sasrec_id": "SASRec (IDs)", "sasrec_text": "SASRec (text features)",
          "sasrec_text_image": "SASRec (text + image features)",
          "sasrec_id_text_image": "SASRec (IDs + text + image)"}
PAPER = {"TIGER": {"recall@5": 0.0454, "ndcg@5": 0.0321, "recall@10": 0.0648, "ndcg@10": 0.0384},
         "SASRec": {"recall@5": 0.0387, "ndcg@5": 0.0249, "recall@10": 0.0605, "ndcg@10": 0.0318}}


def load_runs() -> dict[str, list[dict]]:
    """Experiment name -> one metrics dict per seed."""
    runs = defaultdict(list)
    for path in sorted(RESULTS.glob("**/seed*/metrics.json")):
        m = json.loads(path.read_text())
        runs[m["name"]].append(m)
    return runs


def get(m: dict, path: str):
    for key in path.split("."):
        if not isinstance(m, dict) or key not in m:
            return None
        m = m[key]
    return m


def cell(seeds: list[dict] | None, path: str, digits: int = 4) -> str:
    """Mean over seeds, with the standard deviation when there is more than one seed."""
    values = [v for v in (get(m, path) for m in seeds or []) if v is not None]
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def n_seeds(runs, names) -> str:
    counts = sorted({len(runs[n]) for n in names if n in runs})
    return "/".join(map(str, counts)) if counts else "0"


def main_tables(runs) -> list[str]:
    out = []
    for d, dname in DATASETS.items():
        for s, sname in SPLITS.items():
            names = [f"{d}_{s}_{v}" for v in VARIANTS]
            if not any(n in runs for n in names):
                continue
            header = ["Semantic ID built from", "Recall@5", "NDCG@5", "Recall@10", "NDCG@10",
                      "Recall@10 (tail)", "Recall@10 (mid)", "Recall@10 (head)"]
            if s == "time":
                header.append("Recall@10 (cold)")
            rows = []
            for v, label in VARIANTS.items():
                r = runs.get(f"{d}_{s}_{v}")
                if not r:
                    continue
                row = [label] + [cell(r, f"test.{k}") for k in ("recall@5", "ndcg@5", "recall@10", "ndcg@10")]
                row += [cell(r, f"test.slices.pop/{b}.recall@10") for b in ("tail", "mid", "head")]
                if s == "time":
                    row.append(cell(r, "test.slices.pop/cold.recall@10"))
                rows.append(row)
            out += [f"### ID variants: {dname}, {sname} (seeds: {n_seeds(runs, names)})", "",
                    table(header, rows), ""]
    return out


def best_generative(runs, d: str, s: str) -> str | None:
    """Chosen on validation NDCG@10, not on test."""
    scored = [(np.mean([m["valid"]["ndcg@10"] for m in runs[f"{d}_{s}_{v}"]]), v)
              for v in VARIANTS if f"{d}_{s}_{v}" in runs]
    return max(scored)[1] if scored else None


def baseline_tables(runs) -> list[str]:
    out = []
    for d, dname in DATASETS.items():
        for s, sname in SPLITS.items():
            rows = []
            pop = runs.get(f"{d}_{s}_pop")
            if pop:
                for key, label in (("popularity", "Popularity"), ("popularity_recent", "Popularity, last 90 days")):
                    rows.append([label, cell(pop, f"{key}.test.recall@10"), cell(pop, f"{key}.test.ndcg@10"),
                                 "–" if s != "time" else cell(pop, f"{key}.test.slices.pop/cold.recall@10"),
                                 "–", "–", "–"])
            entries = [(f"{d}_{s}_{k}", label) for k, label in SASREC.items()]
            best = best_generative(runs, d, s)
            if best:
                entries.append((f"{d}_{s}_{best}", f"Generative, best semantic ID ({best})"))
            for name, label in entries:
                r = runs.get(name)
                if r:
                    rows.append([label, cell(r, "test.recall@10"), cell(r, "test.ndcg@10"),
                                 "–" if s != "time" else cell(r, "test.slices.pop/cold.recall@10"),
                                 cell(r, "train_gpu_hours", 2), cell(r, "test.ms_per_user", 2),
                                 cell(r, "peak_gpu_gb", 1)])
            if rows:
                out += [f"### Baselines: {dname}, {sname}", "",
                        table(["Model", "Recall@10", "NDCG@10", "Recall@10 (cold)", "Train GPU-hours",
                               "Inference ms/user", "Peak GPU GB"], rows), ""]
    return out


def reproduction_table(runs) -> list[str]:
    rows = []
    for label, name in (("TIGER", "beauty2014_tiger"), ("SASRec", "beauty2014_sasrec")):
        r = runs.get(name)
        if not r:
            continue
        row = [label]
        for k in ("recall@5", "ndcg@5", "recall@10", "ndcg@10"):
            row += [f"{PAPER[label][k]:.4f}", cell(r, f"test.{k}")]
        rows.append(row)
    if not rows:
        return []
    header = ["Beauty 2014"]
    for k in ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10"):
        header += [f"{k} (paper)", f"{k} (mine)"]
    return ["### Reproduction on Amazon Beauty 2014", "", table(header, rows), ""]


def slice_tables(runs) -> list[str]:
    out = []
    groups = [("history/under_5", "History < 5"), ("history/5_to_10", "History 5-10"),
              ("history/over_10", "History > 10"), ("image/real", "Target has an image"),
              ("image/stand_in", "Target has no image")]
    for d, dname in DATASETS.items():
        for s, sname in SPLITS.items():
            rows = []
            for key, label in list(VARIANTS.items()) + [(k, v) for k, v in SASREC.items()]:
                r = runs.get(f"{d}_{s}_{key}")
                if r:
                    rows.append([label] + [cell(r, f"test.slices.{g}.ndcg@10") for g, _ in groups])
            if rows:
                counts = [str(get(runs[f"{d}_{s}_{k}"][0], f"test.slices.{g}.n") or 0)
                          for k in VARIANTS if f"{d}_{s}_{k}" in runs for g, _ in groups][: len(groups)]
                out += [f"### NDCG@10 by slice: {dname}, {sname}", "",
                        table(["Model"] + [f"{label} (n={n})" for (_, label), n in zip(groups, counts, strict=False)], rows), ""]
    return out


def behaviour_tables(runs) -> list[str]:
    out = []
    for d, dname in DATASETS.items():
        for s, sname in SPLITS.items():
            rows = []
            for key, label in list(VARIANTS.items()) + [("sasrec_id", SASREC["sasrec_id"]),
                                                        ("sasrec_text_image", SASREC["sasrec_text_image"])]:
                r = runs.get(f"{d}_{s}_{key}")
                if not r:
                    continue
                pm = get(r[0], "test.prefix_match")
                rows.append([label,
                             " / ".join(f"{x:.3f}" for x in pm) if pm else "–",
                             cell(r, "test.invalid_beam_share_unconstrained", 3),
                             cell(r, "test.diversity.coverage@10", 3),
                             cell(r, "test.diversity.category_entropy@10", 2),
                             cell(r, "test.diversity.mean_popularity@10", 1)])
            if rows:
                out += [f"### Behaviour: {dname}, {sname}", "",
                        table(["Model", "Top beam has first 1 / 2 / ... tokens right",
                               "Invalid beams, unconstrained", "Catalogue coverage@10",
                               "Category entropy@10", "Mean popularity of recommendations"], rows), ""]
    return out


def tokenizer_table() -> list[str]:
    rows = []
    for path in sorted((RESULTS / "tokenizer").glob("*/*/*.json")):
        r = json.loads(path.read_text())
        if r["data"] == "all_beauty":
            continue
        rows.append([r["data"], r["split"], r["id"],
                     " / ".join(f"{u:.0%}" for u in r["usage"]),
                     " / ".join(f"{p:.0f}" for p in r["perplexity"]),
                     f"{r['recon_err_train']:.3f}",
                     "–" if r["recon_err_unseen"] is None else f"{r['recon_err_unseen']:.3f}",
                     f"{r['collision_rate']:.1%}", str(r["largest_group"]),
                     f"{r['prefix_purity']:.3f}", f"{r['chance_purity']:.3f}"])
    if not rows:
        return []
    return ["### Tokeniser report", "",
            table(["Data", "Split", "ID", "Codebook usage per level", "Perplexity per level",
                   "Recon. error (train items)", "Recon. error (unseen items)", "Collision rate",
                   "Largest collision group", "Prefix purity", "Chance purity"], rows), ""]


def analysis_section() -> list[str]:
    path = RESULTS / "analysis" / "bootstrap.json"
    if not path.exists():
        return []
    rows = [[r["data"], r["split"], r["a"], r["b"], f"{r['mean_diff']:+.4f}",
             f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", f"{r['p_two_sided']:.3f}"]
            for r in json.loads(path.read_text())]
    return ["### Paired bootstrap over test users, NDCG@10 (A minus B)", "",
            table(["Data", "Split", "A", "B", "Difference", "95% CI", "p"], rows), ""]


def main() -> None:
    runs = load_runs()
    sections = (reproduction_table(runs) + main_tables(runs) + baseline_tables(runs)
                + slice_tables(runs) + behaviour_tables(runs) + analysis_section() + tokenizer_table())
    text = "# Results tables\n\nGenerated by `scripts/make_results_tables.py`. Do not edit.\n\n" + "\n".join(sections)
    (RESULTS / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
