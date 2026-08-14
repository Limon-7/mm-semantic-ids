import numpy as np
import pytest

from mmsid.tokenizer.assign import add_collision_token, level_sizes
from mmsid.tokenizer.fusion import IdSpec, build_content_codes
from mmsid.tokenizer.metrics import collision_stats, perplexity, prefix_purity
from mmsid.tokenizer.rqkmeans import ResidualKMeans


def test_collision_token_numbers_duplicates_in_item_order():
    codes = np.array([[1, 2], [3, 4], [1, 2], [1, 2], [3, 4], [5, 6]])
    ids = add_collision_token(codes)
    assert ids[:, -1].tolist() == [0, 0, 1, 2, 1, 0]
    assert len(np.unique(ids, axis=0)) == len(ids)
    assert level_sizes(ids, [8, 8]) == [8, 8, 3]


def test_collision_stats():
    stats = collision_stats(np.array([[1, 2], [3, 4], [1, 2], [1, 2]]))
    assert stats["largest_group"] == 3 and stats["distinct_codes"] == 2
    assert stats["collision_rate"] == pytest.approx(0.5)


def test_perplexity_uniform():
    assert perplexity(np.array([0, 1, 2, 3] * 5)) == pytest.approx(4.0)


def test_prefix_purity_perfect_and_chance():
    codes = np.array([[0, 0, 1], [0, 0, 2], [1, 1, 0], [1, 1, 3]])
    out = prefix_purity(codes, np.array(["a", "a", "b", "b"]))
    assert out["prefix_purity"] == pytest.approx(1.0)
    assert out["chance_purity"] == pytest.approx(4 / 12)


def test_residual_kmeans_recovers_clusters():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 8)) * 10
    x = np.concatenate([c + 0.01 * rng.normal(size=(50, 8)) for c in centres]).astype(np.float32)
    codes1, err1 = ResidualKMeans(1, 4, seed=0).fit(x).encode(x)
    codes2, err2 = ResidualKMeans(2, 4, seed=0).fit(x).encode(x)
    assert (codes1[:, 0] == codes2[:, 0]).all()      # the first level does not depend on depth
    assert err2.mean() < err1.mean()                 # each level removes part of the residual
    # two items from different clusters never share a full code
    truth = np.repeat(np.arange(4), 50)
    full = codes2[:, 0] * 4 + codes2[:, 1]
    assert all(len(set(truth[full == c])) == 1 for c in set(full))


@pytest.mark.parametrize("variant", ["T", "I", "EF", "LF"])
def test_every_variant_gives_the_same_id_length(variant):
    rng = np.random.default_rng(1)
    text = rng.normal(size=(300, 16)).astype(np.float32)
    image = rng.normal(size=(300, 12)).astype(np.float32)
    mask = np.arange(300) < 250
    spec = IdSpec(variant=variant, quantizer="rqkmeans", levels=4, codebook_size=8)
    codes, err, fitted = build_content_codes(spec, text, image, mask, qcfg={"niter": 5})
    assert codes.shape == (300, 4) and codes.max() < 8 and err.shape == (300,)
    expected = {"T": ["text"], "I": ["image"], "EF": ["joint"], "LF": ["text", "image"]}
    assert [m for m, _ in fitted] == expected[variant]


def test_id_names():
    assert IdSpec(variant="LF").name == "LF_rqvae_4x256_t2i2"
    assert IdSpec(variant="LF", lf_order="image_first").name == "LF_rqvae_4x256_t2i2_imgfirst"
    assert IdSpec(variant="T", levels=3).name == "T_rqvae_3x256"
