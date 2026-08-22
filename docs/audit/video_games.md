# Data audit: `video_games`

Source category: `Video_Games`, 5-core benchmark files.

| | |
|---|---|
| Users | 94,762 |
| Items | 25,612 |
| Interactions | 814,586 |
| Density | 0.03356% |
| Items found in metadata | 100.0% |
| Items with an image URL | 100.0% |
| Items with a downloaded image | 100.0% |
| Items with a non-empty title | 100.0% |
| Distinct leaf categories | 136 |

## Sequence length (interactions per user)

| Length | Users | Share |
|---|---|---|
| 5 | 30,451 | 32.1% |
| 6-10 | 47,241 | 49.9% |
| 11-20 | 13,180 | 13.9% |
| 21-50 | 3,417 | 3.6% |
| 51+ | 473 | 0.5% |

Mean 8.6, median 6, max 473.

## Item popularity (training interactions, leave-last-out)

- Median 9, mean 24.4, max 2,062.
- Top 1% of items hold 18.9% of training interactions, top 10% hold 56.8%.
- Gini coefficient 0.657.

## Splits

| Split | Train rows | Valid rows | Test rows | Items seen in training | Cold share (valid) | Cold share (test) | Target already in history (test) |
|---|---|---|---|---|---|---|---|
| `last_out` | 530,300 | 94,762 | 94,762 | 25,527 | 0.2% | 0.4% | 0.00% |
| `timestamp` | 645,265 | 33,094 | 41,465 | 22,976 | 27.9% | 60.3% | 0.00% |

Rows are `(history, target)` pairs with a non-empty history. A cold target is an item with no interaction in that split's training file.
