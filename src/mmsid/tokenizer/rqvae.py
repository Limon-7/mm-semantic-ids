"""RQ-VAE as described in the TIGER paper.

An MLP encoder maps the item embedding to a small latent, a stack of codebooks quantises the
latent residual by residual, and a mirrored decoder reconstructs the embedding.
"""

import numpy as np
import torch
from torch import nn

from mmsid.tokenizer.rqkmeans import kmeans, nearest


def mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], code_dim: int, n_levels: int,
                 codebook_size: int, beta: float = 0.25):
        super().__init__()
        self.n_levels, self.codebook_size, self.beta = n_levels, codebook_size, beta
        self.encoder = mlp([in_dim, *hidden, code_dim])
        self.decoder = mlp([code_dim, *reversed(hidden), in_dim])
        self.codebooks = nn.Parameter(torch.randn(n_levels, codebook_size, code_dim) * 0.01)

    def quantise(self, z: torch.Tensor):
        residual, quantised = z, torch.zeros_like(z)
        codes, residuals, loss = [], [], z.new_zeros(())
        for level in range(self.n_levels):
            cb = self.codebooks[level]
            idx = torch.cdist(residual, cb).argmin(1)
            e = cb[idx]
            # codebook term pulls the code to the residual, commitment term does the reverse
            loss = loss + ((e - residual.detach()) ** 2).sum(1).mean()
            loss = loss + self.beta * ((residual - e.detach()) ** 2).sum(1).mean()
            residuals.append(residual.detach())
            quantised = quantised + e
            residual = residual - e.detach()
            codes.append(idx)
        z_q = z + (quantised - z).detach()  # straight-through estimator
        return z_q, torch.stack(codes, 1), loss, residuals

    def forward(self, x: torch.Tensor):
        z_q, codes, q_loss, residuals = self.quantise(self.encoder(x))
        recon = self.decoder(z_q)
        recon_loss = ((recon - x) ** 2).sum(1).mean()
        return recon, codes, recon_loss, q_loss, residuals

    @torch.no_grad()
    def init_codebooks(self, x: torch.Tensor, seed: int) -> None:
        """k-means initialisation, level by level, on the encoder output of a data sample."""
        residual = self.encoder(x).cpu().numpy()
        for level in range(self.n_levels):
            cb = kmeans(residual, self.codebook_size, seed + level, niter=20)
            residual = residual - cb[nearest(residual, cb)]
            self.codebooks[level].copy_(torch.from_numpy(cb))

    @torch.no_grad()
    def restart_dead_codes(self, used: torch.Tensor, residuals: list[torch.Tensor]) -> int:
        """Move codes that were never chosen onto residuals of random items in the batch."""
        n_dead = 0
        for level in range(self.n_levels):
            dead = (~used[level]).nonzero().flatten()
            if len(dead):
                pick = torch.randint(len(residuals[level]), (len(dead),), device=dead.device)
                self.codebooks[level, dead] = residuals[level][pick]
                n_dead += len(dead)
        return n_dead


class RQVAEQuantizer:
    """fit/encode wrapper with the same interface as ResidualKMeans."""

    def __init__(self, n_levels: int, codebook_size: int, seed: int = 0, hidden=(512, 256, 128),
                 code_dim: int = 32, beta: float = 0.25, steps: int = 20000, batch_size: int = 1024,
                 optimizer: str = "adamw", lr: float = 1e-3, restart_every: int = 500,
                 device: str | None = None):
        self.n_levels, self.codebook_size, self.seed = n_levels, codebook_size, seed
        self.hidden, self.code_dim, self.beta = list(hidden), code_dim, beta
        self.steps, self.batch_size = steps, batch_size
        self.optimizer, self.lr, self.restart_every = optimizer, lr, restart_every
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: RQVAE | None = None
        self.log: list[dict] = []

    def fit(self, x: np.ndarray) -> "RQVAEQuantizer":
        torch.manual_seed(self.seed)
        data = torch.from_numpy(x.astype(np.float32)).to(self.device)
        model = RQVAE(data.shape[1], self.hidden, self.code_dim, self.n_levels,
                      self.codebook_size, self.beta).to(self.device)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        sample = data[torch.randperm(len(data), device=self.device, generator=gen)[:50000]]
        model.init_codebooks(sample, self.seed)

        if self.optimizer == "adagrad":
            opt = torch.optim.Adagrad(model.parameters(), lr=self.lr, initial_accumulator_value=0.1)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr)

        used = torch.zeros(self.n_levels, self.codebook_size, dtype=torch.bool, device=self.device)
        perm, ptr = torch.randperm(len(data), device=self.device, generator=gen), 0
        for step in range(1, self.steps + 1):
            if ptr + self.batch_size > len(perm):
                perm, ptr = torch.randperm(len(data), device=self.device, generator=gen), 0
            batch = data[perm[ptr : ptr + self.batch_size]]
            ptr += self.batch_size

            _, codes, recon_loss, q_loss, residuals = model(batch)
            opt.zero_grad(set_to_none=True)
            (recon_loss + q_loss).backward()
            opt.step()

            used.scatter_(1, codes.T, True)
            if step % self.restart_every == 0:
                n_dead = model.restart_dead_codes(used, residuals) if step < self.steps else 0
                self.log.append({"step": step, "recon": recon_loss.item(), "quant": q_loss.item(),
                                 "usage": used.float().mean().item(), "restarted": n_dead})
                used.zero_()
        self.model = model.eval()
        return self

    @torch.no_grad()
    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        codes, errs = [], []
        for i in range(0, len(x), 8192):
            batch = torch.from_numpy(x[i : i + 8192].astype(np.float32)).to(self.device)
            recon, c, _, _, _ = self.model(batch)
            err = ((recon - batch) ** 2).sum(1) / (batch**2).sum(1).clamp_min(1e-12)
            codes.append(c.cpu().numpy())
            errs.append(err.cpu().numpy())
        return np.concatenate(codes), np.concatenate(errs)
