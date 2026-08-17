import numpy as np
import pytest

from mmsid.eval.metrics import ndcg_at_k, paired_bootstrap, recall_at_k, summarise, target_ranks
from mmsid.eval.slices import history_bucket, popularity_bucket


def test_target_ranks():
    rec = np.array([[5, 3, 9], [1, 2, 3], [7, 7, 7]])
    assert target_ranks(rec, np.array([3, 4, 7])).tolist() == [1, -1, 0]


def test_recall_and_ndcg_by_hand():
    ranks = np.array([0, 1, 4, 9, -1])
    assert recall_at_k(ranks, 5).tolist() == [1, 1, 1, 0, 0]
    assert recall_at_k(ranks, 10).tolist() == [1, 1, 1, 1, 0]
    expected = [1.0, 1 / np.log2(3), 1 / np.log2(6), 0.0, 0.0]
    assert ndcg_at_k(ranks, 5) == pytest.approx(expected)
    out = summarise(ranks, (5, 10))
    assert out["recall@5"] == pytest.approx(0.6)
    assert out["ndcg@10"] == pytest.approx((1 + 1 / np.log2(3) + 1 / np.log2(6) + 1 / np.log2(11)) / 5)


def test_summarise_empty_slice():
    assert summarise(np.array([], dtype=int))["recall@10"] is None


def test_paired_bootstrap_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    b = rng.random(2000) < 0.1
    a = b | (rng.random(2000) < 0.1)
    out = paired_bootstrap(a.astype(float), b.astype(float), n_boot=500)
    assert out["ci_low"] > 0 and out["mean_diff"] == pytest.approx(a.mean() - b.mean())


def test_popularity_bucket():
    pop = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # index 0 is padding
    bucket = popularity_bucket(pop)
    assert bucket[1] == 0                      # cold
    assert bucket[2:7].tolist() == [1] * 5     # bottom half of the 10 warm items
    assert bucket[7:10].tolist() == [2] * 3
    assert bucket[10:].tolist() == [3] * 2


def test_history_bucket():
    assert history_bucket(np.array([1, 4, 5, 10, 11, 20])).tolist() == [0, 0, 1, 1, 2, 2]
