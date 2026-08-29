# Implementation notes

Decisions that are not obvious from the code, and where I left the TIGER paper's settings.

## RQ-VAE optimiser

The paper trains the RQ-VAE with Adagrad at learning rate 0.4. In my PyTorch code that diverges:
after 5,000 steps on Beauty 2014 the relative reconstruction error was in the thousands, even with
TensorFlow's initial accumulator value of 0.1. AdamW at 1e-3 with the same architecture and
loss trains cleanly (5,000 steps on Beauty 2014: reconstruction error 0.087, codebook usage
100% / 100% / 98%, 0.9% collisions). Every RQ-VAE in this repo uses AdamW at 1e-3 for 20,000 steps.
Adagrad is still selectable with `ids.rqvae.optimizer=adagrad`.

The codebooks are initialised with k-means on the encoder output of up to 50,000 training items
(the paper uses the first batch; 1,024 points are too few for 256 centroids). Codes that are not
chosen for 500 steps are moved onto the residual of a random item.

## Sequence model optimiser and budget

The paper trains for 200k steps at batch 256 with learning rate 0.01 and inverse-square-root decay.
I use AdamW at 1e-3 with 2k warm-up steps and cosine decay, the common setting in open
re-implementations, and a smaller budget: 60k steps on Beauty 2014 and at most 40k steps on the
2023 categories, with early stopping on validation NDCG@10 (10k sampled validation users, beam 20,
every 2k steps, patience 5). Final numbers use the best checkpoint and beam 50.

The reason for the smaller budget is the machine. The server has an old CPU, and with a model
this small a training step is bound by Python and kernel-launch overhead, about 0.12 s per step
whatever the GPU is doing. 200k steps would be around 7 hours per run.

## Length-bucketed batches

Histories are padded to 20 items, which is 100 tokens with 5 tokens per item, yet the median user
has 6 to 7 interactions. Training batches are drawn from buckets of similar history length and
trimmed to the longest history in the batch. A bucket is picked with probability proportional to
its size, so each row is still equally likely to be sampled. Evaluation sorts rows by history
length for the same reason and restores the order afterwards. T5 uses relative positions and the
padding is masked, so trimming does not change any output; SASRec counts its positions from the
newest item for the same reason. Both are covered by tests.

## Constrained beam search

Decoding is my own beam search, not `generate()` with `prefix_allowed_tokens_fn`. The trie of valid
IDs is flattened into one transition table per level (`models/trie.py`), so masking a whole batch
of beams is a single tensor lookup instead of a Python call per beam per step. A test checks that,
when the beam is wide enough to hold every item, the search returns exactly the item with the
highest teacher-forced log-probability.

Each ID position has its own token range and the decoder is only read inside the range of the
position being decoded. "Unconstrained" decoding in the invalid-beam analysis therefore still knows
which level it is on; it only ignores which combinations exist.

## Evaluation choices

- Ranking is always against the full catalogue. For the generative model that means the top 10 of
  50 beams over valid IDs.
- History items are not removed from the lists, for any model (see the data audit).
- The user token (2,000 hashed users, as in the paper) is on. I did not ablate it.
- Popularity buckets: cold = no training interaction; among the rest, the bottom 50% of items by
  training popularity are tail, the next 30% mid, the top 20% head.
- SASRec uses the same `(history, target)` rows as the generative model and predicts from the last
  position with a full-softmax cross-entropy loss. Dimension and dropout are picked on a small
  validation grid for the ID variant (`scripts/tune_sasrec.sh`) and reused for the content variants.

## What I did not run

One seed (0) for everything, so there are no standard deviations; the paired bootstrap over test
users is the only uncertainty estimate. No ablations over the quantiser, ID shape, text encoder or
late-fusion order. Residual k-means is implemented and tested but all reported IDs use RQ-VAE.

## Results of the SASRec grid

Validation NDCG@10 on the leave-last-out split, ID variant, 30k steps. The best setting per
category is the default in `scripts/run_matrix.sh`.

| Dimension | Dropout | `video_games` | `toys_and_games` |
|---|---|---|---|
| 64 | 0.1 | 0.0484 | 0.0069 |
| 64 | 0.2 | 0.0490 | 0.0062 |
| 64 | 0.5 | 0.0382 | 0.0026 |
| 128 | 0.1 | 0.0433 | 0.0050 |
| 128 | 0.2 | 0.0486 | 0.0068 |
| 128 | 0.5 | 0.0395 | 0.0043 |

## Evaluation memory

With 50 beams the T5 decoder's key/value cache held about 1 GB per layer, and the final evaluation
peaked near 8 GB. On a shared card that was enough to kill runs that had finished training. The
beam search now runs with `use_cache=False` (the decoder input is at most five tokens, so nothing is
lost) and an evaluation batch of 64. Runs evaluated before the change report a higher peak memory
than runs evaluated after it; the metrics are unaffected.
