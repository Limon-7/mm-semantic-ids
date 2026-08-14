"""The four ways of building a semantic ID from text and image embeddings.

    T   text only            all levels quantise the text embedding
    I   image only           all levels quantise the image embedding
    EF  early fusion         concatenate the two embeddings, then quantise the joint vector
    LF  late fusion          some levels quantise text, the others quantise the image

Every variant gives the same number of content tokens, so sequence length is equal.
"""

from dataclasses import dataclass

import numpy as np

from mmsid.tokenizer.rqkmeans import ResidualKMeans
from mmsid.tokenizer.rqvae import RQVAEQuantizer

VARIANTS = ("T", "I", "EF", "LF")


@dataclass
class IdSpec:
    variant: str = "T"
    quantizer: str = "rqvae"
    levels: int = 4
    codebook_size: int = 256
    lf_text_levels: int = 2  # late fusion: how many of the levels go to text
    lf_order: str = "text_first"

    def __post_init__(self):
        assert self.variant in VARIANTS, self.variant
        assert self.quantizer in ("rqvae", "rqkmeans"), self.quantizer

    @property
    def name(self) -> str:
        tag = f"{self.variant}_{self.quantizer}_{self.levels}x{self.codebook_size}"
        if self.variant == "LF":
            a = self.lf_text_levels
            tag += f"_t{a}i{self.levels - a}" + ("" if self.lf_order == "text_first" else "_imgfirst")
        return tag


def plan(spec: IdSpec, text: np.ndarray | None, image: np.ndarray | None):
    """List of (modality name, input matrix, number of levels), in ID order."""
    if spec.variant == "T":
        return [("text", text, spec.levels)]
    if spec.variant == "I":
        return [("image", image, spec.levels)]
    if spec.variant == "EF":
        # both inputs are unit-norm, so the joint vector is unit-norm after the rescale
        return [("joint", np.concatenate([text, image], axis=1) / np.sqrt(2.0), spec.levels)]
    parts = [("text", text, spec.lf_text_levels), ("image", image, spec.levels - spec.lf_text_levels)]
    return parts if spec.lf_order == "text_first" else parts[::-1]


def make_quantizer(spec: IdSpec, n_levels: int, seed: int, qcfg: dict):
    if spec.quantizer == "rqkmeans":
        return ResidualKMeans(n_levels, spec.codebook_size, seed=seed, **qcfg)
    return RQVAEQuantizer(n_levels, spec.codebook_size, seed=seed, **qcfg)


def build_content_codes(spec: IdSpec, text: np.ndarray | None, image: np.ndarray | None,
                        train_mask: np.ndarray, seed: int = 0, qcfg: dict | None = None):
    """Fit on training-period items only, then assign codes to every item.

    text, image: [n_items, d] unit-norm arrays (no padding row). train_mask: [n_items] bool.
    Returns codes [n_items, levels], the mean relative reconstruction error per item, and the
    fitted quantisers.
    """
    codes, errs, fitted = [], [], []
    for i, (modality, x, n_levels) in enumerate(plan(spec, text, image)):
        x = x.astype(np.float32)
        q = make_quantizer(spec, n_levels, seed + 100 * i, dict(qcfg or {})).fit(x[train_mask])
        c, e = q.encode(x)
        codes.append(c)
        errs.append(e)
        fitted.append((modality, q))
    return np.concatenate(codes, axis=1), np.mean(errs, axis=0), fitted
