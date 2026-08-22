# Data audit: `toys_and_games`

Source category: `Toys_and_Games`, 5-core benchmark files.
Users subsampled: 100,000 of 432,264 (seed 0).

| | |
|---|---|
| Users | 100,000 |
| Items | 147,458 |
| Interactions | 891,152 |
| Density | 0.00604% |
| Items found in metadata | 100.0% |
| Items with an image URL | 100.0% |
| Items with a downloaded image | 100.0% |
| Items with a non-empty title | 100.0% |
| Distinct leaf categories | 784 |

## Sequence length (interactions per user)

| Length | Users | Share |
|---|---|---|
| 5 | 30,379 | 30.4% |
| 6-10 | 50,128 | 50.1% |
| 11-20 | 14,885 | 14.9% |
| 21-50 | 4,003 | 4.0% |
| 51+ | 605 | 0.6% |

Mean 8.9, median 7, max 984.

## Item popularity (training interactions, leave-last-out)

- Median 2, mean 4.7, max 809.
- Top 1% of items hold 16.8% of training interactions, top 10% hold 48.9%.
- Gini coefficient 0.599.

## Splits

| Split | Train rows | Valid rows | Test rows | Items seen in training | Cold share (valid) | Cold share (test) | Target already in history (test) |
|---|---|---|---|---|---|---|---|
| `last_out` | 591,152 | 100,000 | 100,000 | 136,782 | 6.7% | 8.4% | 0.00% |
| `timestamp` | 622,748 | 68,851 | 99,553 | 118,577 | 31.6% | 66.7% | 0.00% |

Rows are `(history, target)` pairs with a non-empty history. A cold target is an item with no interaction in that split's training file.
