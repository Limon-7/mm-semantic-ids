"""Residual k-means: k-means on the embeddings, then on what is left over, level by level."""

import faiss
import numpy as np


def kmeans(x: np.ndarray, k: int, seed: int, niter: int = 30) -> np.ndarray:
    km = faiss.Kmeans(x.shape[1], k, niter=niter, seed=seed, verbose=False,
                      min_points_per_centroid=1, max_points_per_centroid=10**9)
    km.train(np.ascontiguousarray(x, dtype=np.float32))
    return km.centroids.copy()


def nearest(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    return index.search(np.ascontiguousarray(x, dtype=np.float32), 1)[1][:, 0]


class ResidualKMeans:
    def __init__(self, n_levels: int, codebook_size: int, seed: int = 0, niter: int = 30):
        self.n_levels, self.codebook_size, self.seed, self.niter = n_levels, codebook_size, seed, niter
        self.codebooks: list[np.ndarray] = []

    def fit(self, x: np.ndarray) -> "ResidualKMeans":
        residual = x.astype(np.float32).copy()
        self.codebooks = []
        for level in range(self.n_levels):
            cb = kmeans(residual, self.codebook_size, self.seed + level, self.niter)
            residual -= cb[nearest(residual, cb)]
            self.codebooks.append(cb)
        return self

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return codes [n, levels] and the relative reconstruction error of each row."""
        residual = x.astype(np.float32).copy()
        codes = np.zeros((len(x), self.n_levels), np.int64)
        for level, cb in enumerate(self.codebooks):
            codes[:, level] = nearest(residual, cb)
            residual -= cb[codes[:, level]]
        err = (residual**2).sum(1) / np.clip((x.astype(np.float32) ** 2).sum(1), 1e-12, None)
        return codes, err
