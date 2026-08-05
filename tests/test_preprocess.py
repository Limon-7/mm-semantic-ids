import os

import numpy as np
import pandas as pd
import pytest

from mmsid.data.preprocess import (
    T1,
    collect_items,
    encode_split,
    item_popularity,
    item_text,
    subsample_users,
)


def _frame():
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "parent_asin": ["A", "B", "C", "B", "D"],
            "timestamp": [1, 2, 3, 5, 6],
            "history": ["", "A", "A B", "", "B"],
        }
    )


def test_collect_items_includes_history_only_items():
    df = _frame().iloc[[2]]  # target C, history A B
    assert collect_items({("s", "p"): df}) == ["A", "B", "C"]


def test_encode_split_drops_empty_history_and_keeps_order():
    df = _frame()
    item2idx = {a: i + 1 for i, a in enumerate("ABCD")}
    enc = encode_split(df, item2idx, {"u1": 0, "u2": 1}, max_history=20)
    assert len(enc) == 3
    assert enc["history"].tolist() == [[1], [1, 2], [2]]
    assert enc["target"].tolist() == [2, 3, 4]
    assert 0 not in enc["target"].tolist()  # index 0 is padding


def test_history_truncated_to_most_recent():
    df = pd.DataFrame(
        {"user_id": ["u"], "parent_asin": ["D"], "timestamp": [9], "history": ["A B C"]}
    )
    enc = encode_split(df, {a: i + 1 for i, a in enumerate("ABCD")}, {"u": 0}, max_history=2)
    assert enc["history"].tolist() == [[2, 3]]


def test_item_popularity_counts_first_interactions():
    pop = item_popularity(_frame(), {a: i + 1 for i, a in enumerate("ABCD")})
    assert pop.tolist() == [0, 1, 2, 1, 1]


def test_subsample_users_is_deterministic():
    users = np.array([f"u{i}" for i in range(100)])
    a = subsample_users(users, 10, seed=0)
    b = subsample_users(users[::-1], 10, seed=0)
    assert len(a) == 10 and (a == b).all()
    assert len(subsample_users(users, None, 0)) == 100


def test_item_text():
    text = item_text("Lego Set", "LEGO", ["Toys", "Building", "Sets"], 19.5, ["a", "b", "c", "d"])
    assert text == "Lego Set. Brand: LEGO. Category: Building > Sets. Price: $19.50. a b c"


# Checks on the real processed data. Skipped when it is not on this machine.
DATASETS = ("all_beauty", "video_games", "toys_and_games")


def _processed(name):
    root = os.environ.get("RS_DATA")
    path = None if not root else os.path.join(root, "processed", name)
    if not path or not os.path.exists(os.path.join(path, "items.parquet")):
        pytest.skip(f"{name} is not processed on this machine")
    return path


@pytest.mark.parametrize("name", DATASETS)
@pytest.mark.parametrize("split", ("last_out", "timestamp"))
def test_real_split_has_no_leak(name, split):
    root = _processed(name)
    parts = {p: pd.read_parquet(os.path.join(root, split, f"{p}.parquet")) for p in ("train", "valid", "test")}
    n_items = len(pd.read_parquet(os.path.join(root, "items.parquet"), columns=["item_idx"]))
    for df in parts.values():
        assert df["target"].between(1, n_items).all()
        assert df["history"].map(len).between(1, 20).all()
    if split == "timestamp":
        assert parts["train"]["timestamp"].max() < T1
        assert parts["valid"]["timestamp"].min() >= T1
        assert parts["test"]["timestamp"].min() > parts["valid"]["timestamp"].max()
    else:
        # leave-last-out: the test target of a user is later than everything in training
        last_train = parts["train"].groupby("user_idx")["timestamp"].max()
        test = parts["test"].set_index("user_idx")["timestamp"]
        common = last_train.index.intersection(test.index)
        assert (test.loc[common] >= last_train.loc[common]).all()
