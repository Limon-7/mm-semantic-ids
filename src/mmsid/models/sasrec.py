"""SASRec (Kang and McAuley, 2018) with a full-softmax cross-entropy loss.

`item_input` chooses what an item is made of:

    id              a learned embedding per item (the classic model)
    text            an MLP on the frozen text embedding
    text_image      an MLP on the frozen text and image embeddings, concatenated
    id_text_image   the sum of the learned embedding and the content MLP

The same item representation is used on the input side and as the output scoring matrix, so
the content variants can score items they never saw in training.
"""

import torch
from torch import nn

ITEM_INPUTS = ("id", "text", "text_image", "id_text_image")


class SASRec(nn.Module):
    def __init__(self, n_items: int, max_len: int, dim: int = 128, layers: int = 2, heads: int = 2,
                 dropout: float = 0.2, item_input: str = "id", content: torch.Tensor | None = None):
        super().__init__()
        assert item_input in ITEM_INPUTS, item_input
        self.item_input = item_input
        self.use_id = item_input in ("id", "id_text_image")
        self.use_content = item_input != "id"

        if self.use_id:
            self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
            nn.init.normal_(self.item_emb.weight, std=0.02)
        if self.use_content:
            assert content is not None and len(content) == n_items + 1
            self.register_buffer("content", content, persistent=False)
            self.content_mlp = nn.Sequential(
                nn.Linear(content.shape[1], 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, dim)
            )

        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        block = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1),
                             persistent=False)

    def item_matrix(self) -> torch.Tensor:
        """[n_items + 1, dim] representation of every item; row 0 is padding."""
        out = 0
        if self.use_id:
            out = out + self.item_emb.weight
        if self.use_content:
            out = out + self.content_mlp(self.content)
        return out

    def user_state(self, history: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """history: [B, T] left-padded with 0. Returns the state after the newest item."""
        T = history.shape[1]
        pad = history == 0
        # positions are counted from the right, so trimming left padding changes nothing
        first = self.pos_emb.num_embeddings - T
        x = items[history] + self.pos_emb(torch.arange(first, first + T, device=history.device))
        x = self.drop(x).masked_fill(pad[:, :, None], 0.0)
        # padding sits on the left, so a padded query position would have nothing to attend to
        # under the causal mask; letting it see itself avoids NaNs and never reaches real items
        mask = self.causal[:T, :T][None] | pad[:, None, :]
        mask = mask & ~torch.eye(T, dtype=torch.bool, device=history.device)[None]
        mask = mask.repeat_interleave(self.encoder.layers[0].self_attn.num_heads, 0)
        h = self.encoder(x, mask=mask)
        return self.norm(h[:, -1])

    def scores(self, history: torch.Tensor) -> torch.Tensor:
        items = self.item_matrix()
        logits = self.user_state(history, items) @ items.T
        logits[:, 0] = -torch.inf
        return logits

    def forward(self, user, history, target) -> torch.Tensor:
        return nn.functional.cross_entropy(self.scores(history), target)

    @torch.no_grad()
    def recommend(self, user, history, k: int = 10):
        top = self.scores(history).float().topk(k, dim=1)
        return top.indices, top.values
