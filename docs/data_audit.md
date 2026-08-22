# Data audit

Everything here was measured by `python -m mmsid.data.audit` on the processed data. The full
per-dataset reports are in [audit/](audit/).

## Source

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab), Hugging Face dataset
`McAuley-Lab/Amazon-Reviews-2023`. I use the 5-core benchmark files, which exist in two versions:

- `benchmark/5core/last_out_w_his/{Category}.{train,valid,test}.csv`: leave-last-out. The last item
  of each user is the test target, the second-last is validation.
- `benchmark/5core/timestamp_w_his/{Category}.{train,valid,test}.csv`: absolute time split. Training
  is everything before `t1 = 1628643414042` ms, validation is `[t1, t2)`, test is from
  `t2 = 1658002729837` ms.

Item metadata comes from `raw/meta_categories/meta_{Category}.jsonl`. Items are keyed by
`parent_asin`. The files are plain `.csv`, not `.csv.gz`.

The dataset card on Hugging Face does not state a licence. I do not redistribute any of it: the
download script fetches the files from the official repository. Citation:

> Hou, Li, He, Yan, Chen, McAuley. "Bridging Language and Items for Retrieval and Recommendation".
> arXiv:2403.03952, 2024.

## Which categories

| Role | Category | 5-core users | 5-core items | Used |
|---|---|---|---|---|
| Smoke test | `All_Beauty` | 253 | 356 | all |
| Specification-driven | `Video_Games` | 94,762 | 25,612 | all |
| Visually driven | `Toys_and_Games` | 432,264 | 162,035 | 100,000 users (seed 0), 147,458 items |

`Toys_and_Games` is above the item budget I set (20K to 150K), so I kept a random sample of 100,000
users and every item those users touched. Sampling users leaves the item side sparser than 5-core:
the median item has 2 training interactions, against 9 in `Video_Games`. `Beauty_and_Personal_Care`
is larger still and I did not use it.

## Summary

| | `video_games` | `toys_and_games` |
|---|---|---|
| Users | 94,762 | 100,000 |
| Items | 25,612 | 147,458 |
| Interactions | 814,586 | 891,152 |
| Items with a downloaded image | 100.0% | 100.0% |
| Distinct leaf categories | 136 | 784 |
| Median interactions per user | 6 | 7 |
| Median training interactions per item | 9 | 2 |
| Cold test targets, leave-last-out | 0.4% | 8.4% |
| Cold test targets, time split | 60.3% | 66.7% |
| Test targets already in the user's history | 0.00% | 0.00% |

What this means for the experiments:

- **Every item has an image.** The stand-in for missing images (SigLIP text tower on the title) is
  implemented and tested but is never used on these two categories, so I cannot report a
  with-image against without-image comparison.
- **The time split is mostly a cold-start problem.** Around 60 to 67% of test targets were never
  seen in training. A model with ID embeddings cannot rank those items at all, so the time split is
  where content-based codes should matter most.
- **No repeat purchases.** The benchmark has no test target that already sits in the user's
  history. I still do not filter history items out of the recommendation lists, for any model, so
  every model is scored under the same rule.
- Rows with an empty history (the first interaction of a user) are dropped from training and
  evaluation because there is nothing to condition on. In the time split this removes 4.1% of test
  rows for `Video_Games` and 2.8% for `Toys_and_Games`.

## Sanity dataset

Amazon Beauty 2014 5-core, the dataset of the TIGER paper, from the original SNAP files: 22,363
users, 12,101 items, 198,502 interactions, which matches the statistics in the paper.
