# Data audit: `all_beauty`

Source category: `All_Beauty`, 5-core benchmark files.

| | |
|---|---|
| Users | 253 |
| Items | 356 |
| Interactions | 2,535 |
| Density | 2.81454% |
| Items found in metadata | 100.0% |
| Items with an image URL | 100.0% |
| Items with a downloaded image | 100.0% |
| Items with a non-empty title | 100.0% |
| Distinct leaf categories | 2 |

## Sequence length (interactions per user)

| Length | Users | Share |
|---|---|---|
| 5 | 59 | 23.3% |
| 6-10 | 127 | 50.2% |
| 11-20 | 48 | 19.0% |
| 21-50 | 17 | 6.7% |
| 51+ | 2 | 0.8% |

Mean 10.0, median 7, max 63.

## Item popularity (training interactions, leave-last-out)

- Median 5, mean 5.7, max 14.
- Top 1% of items hold 2.0% of training interactions, top 10% hold 18.3%.
- Gini coefficient 0.256.

## Splits

| Split | Train rows | Valid rows | Test rows | Items seen in training | Cold share (valid) | Cold share (test) | Target already in history (test) |
|---|---|---|---|---|---|---|---|
| `last_out` | 1,776 | 253 | 253 | 341 | 7.9% | 27.3% | 0.00% |
| `timestamp` | 1,985 | 275 | 22 | 319 | 75.6% | 72.7% | 0.00% |

Rows are `(history, target)` pairs with a non-empty history. A cold target is an item with no interaction in that split's training file.
