import numpy as np
import torch

from mmsid.data.dataset import BucketSampler, pad_histories, trim
from mmsid.models.sasrec import SASRec
from mmsid.models.tiger import Tiger
from mmsid.models.trie import PrefixIndex
from mmsid.tokenizer.assign import add_collision_token, level_sizes, with_padding_row


def _ids(n=60, levels=3, size=4, seed=0):
    rng = np.random.default_rng(seed)
    ids = with_padding_row(add_collision_token(rng.integers(0, size, size=(n, levels))))
    return ids, level_sizes(ids, [size] * levels)


def test_pad_and_trim():
    h = torch.from_numpy(pad_histories([[1, 2], [3], [4, 5, 6]], max_len=5))
    assert h.tolist() == [[0, 0, 0, 1, 2], [0, 0, 0, 0, 3], [0, 0, 4, 5, 6]]
    assert trim(h).tolist() == [[0, 1, 2], [0, 0, 3], [4, 5, 6]]


def test_bucket_sampler_groups_similar_lengths():
    lengths = torch.tensor([1] * 50 + [20] * 50)
    sampler = BucketSampler(lengths, seed=0)
    for _ in range(20):
        idx = sampler.sample(16)
        assert len(set(lengths[idx].tolist())) == 1


def test_prefix_index_matches_brute_force():
    rng = np.random.default_rng(0)
    ids = with_padding_row(add_collision_token(rng.integers(0, 4, size=(200, 3))))
    sizes = level_sizes(ids, [4, 4, 4])
    index = PrefixIndex(ids, sizes)
    for item in range(1, len(ids)):
        assert index.lookup(ids[item].tolist()) == item
    assert index.lookup([0, 0, 0, sizes[-1] + 5]) == 0
    real = ids[1:]
    for prefix in ([], [2], [1, 3], [0, 0, 0]):
        rows = real[(real[:, : len(prefix)] == prefix).all(1)] if prefix else real
        assert index.allowed(prefix) == sorted(set(rows[:, len(prefix)].tolist()))


def test_tiger_token_ranges_do_not_overlap():
    ids, sizes = _ids()
    model = Tiger(ids, sizes, d_model=16, d_ff=32, d_kv=8, heads=2, layers=1, user_tokens=10)
    tokens = model.item_tokens[1:]
    for level in range(len(sizes)):
        lo = model.offsets[level]
        assert ((tokens[:, level] >= lo) & (tokens[:, level] < lo + sizes[level])).all()
    assert model.user_offset == 1 + sum(sizes)
    assert (model.item_tokens[0] == 0).all()


def test_constrained_beam_search_returns_only_real_distinct_items():
    ids, sizes = _ids()
    torch.manual_seed(0)
    model = Tiger(ids, sizes, d_model=16, d_ff=32, d_kv=8, heads=2, layers=1, user_tokens=10).eval()
    index = PrefixIndex(ids, sizes)
    history = torch.from_numpy(pad_histories([[1, 2, 3], [7], [10, 11]], max_len=5))
    items, scores = model.recommend(torch.tensor([0, 1, 2]), history, index, beam=10)
    assert items.shape == (3, 10)
    assert ((items >= 1) & (items <= 60)).all()
    assert all(len(set(row.tolist())) == 10 for row in items)
    assert (scores[:, :-1] >= scores[:, 1:]).all()


def test_beam_search_equals_exhaustive_search_when_beam_covers_everything():
    ids, sizes = _ids(n=12, levels=2, size=3)
    torch.manual_seed(1)
    model = Tiger(ids, sizes, d_model=16, d_ff=32, d_kv=8, heads=2, layers=1, user_tokens=0).eval()
    index = PrefixIndex(ids, sizes)
    history = torch.tensor([[0, 3, 5]])
    user = torch.tensor([0])
    items, scores = model.recommend(user, history, index, beam=200)

    # score every item directly from the decoder, teacher-forced on its own tokens
    tokens, mask = model.encode_inputs(user, history)
    exact = []
    for item in range(1, 13):
        target = model.item_tokens[item][None]
        dec_in = torch.cat([torch.zeros(1, 1, dtype=torch.long), target[:, :-1]], dim=1)
        logp = model.t5(input_ids=tokens, attention_mask=mask,
                        decoder_input_ids=dec_in).logits.log_softmax(-1)
        exact.append(logp[0, torch.arange(target.shape[1]), target[0]].sum().item())
    best = int(np.argmax(exact)) + 1
    assert items[0, 0].item() == best
    assert abs(scores[0, 0].item() - max(exact)) < 1e-4
    assert sorted(items[0][items[0] > 0].tolist()) == list(range(1, 13))


def test_sasrec_is_unchanged_by_trimming_and_never_recommends_padding():
    torch.manual_seed(0)
    content = torch.randn(31, 6)
    for kind in ("id", "text", "id_text_image"):
        model = SASRec(30, 8, dim=16, layers=1, heads=2, item_input=kind, content=content).eval()
        history = torch.from_numpy(pad_histories([[1, 2, 3], [9]], max_len=8))
        full = model.scores(history)
        short = model.scores(trim(history))
        assert torch.allclose(full[:, 1:], short[:, 1:], atol=1e-5)
        top, _ = model.recommend(None, history, k=5)
        assert (top > 0).all()
