"""TIGER-style generative retrieval: an encoder-decoder transformer over semantic-ID tokens.

Token layout: 0 is padding (and the decoder start token), then one block of tokens per ID
position (each level has its own range), then the hashed user tokens.
"""

import numpy as np
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

from mmsid.models.trie import PrefixIndex


class Tiger(nn.Module):
    def __init__(self, ids: np.ndarray, level_sizes: list[int], d_model: int = 128,
                 d_ff: int = 1024, d_kv: int = 64, heads: int = 6, layers: int = 4,
                 dropout: float = 0.1, user_tokens: int = 2000):
        super().__init__()
        self.level_sizes = list(level_sizes)
        self.n_levels = len(level_sizes)
        self.offsets = [1 + int(sum(level_sizes[:i])) for i in range(self.n_levels)]
        self.user_tokens = user_tokens
        self.user_offset = 1 + int(sum(level_sizes))
        vocab = self.user_offset + user_tokens

        # item_idx -> its tokens; row 0 stays all padding
        tokens = torch.from_numpy(ids) + torch.tensor(self.offsets)
        tokens[0] = 0
        self.register_buffer("item_tokens", tokens, persistent=False)

        config = T5Config(
            vocab_size=vocab, d_model=d_model, d_kv=d_kv, d_ff=d_ff, num_layers=layers,
            num_decoder_layers=layers, num_heads=heads, dropout_rate=dropout,
            feed_forward_proj="relu", pad_token_id=0, decoder_start_token_id=0, eos_token_id=1,
        )
        self.t5 = T5ForConditionalGeneration(config)

    def encode_inputs(self, user: torch.Tensor, history: torch.Tensor):
        """[B] users and [B, T] item indices -> token ids and attention mask."""
        tokens = self.item_tokens[history].flatten(1)
        if self.user_tokens:
            user_tok = self.user_offset + user % self.user_tokens
            tokens = torch.cat([user_tok[:, None], tokens], dim=1)
        return tokens, tokens != 0

    def forward(self, user, history, target) -> torch.Tensor:
        tokens, mask = self.encode_inputs(user, history)
        return self.t5(input_ids=tokens, attention_mask=mask, labels=self.item_tokens[target]).loss

    @torch.no_grad()
    def recommend(self, user, history, index: PrefixIndex, beam: int = 50, constrained: bool = True):
        """Beam search over semantic IDs.

        Returns items [B, beam] (item_idx, 0 where a beam is not a real item) and their
        log-probabilities, best first. With constrained=False the search may leave the set of
        valid IDs; those beams come back as item 0.
        """
        tokens, mask = self.encode_inputs(user, history)
        memory = self.t5.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        B = len(user)

        decoded = torch.zeros(B, 1, 1, dtype=torch.long, device=tokens.device)  # start token
        scores = torch.zeros(B, 1, device=tokens.device)
        prefix = torch.zeros(B, 1, dtype=torch.long, device=tokens.device)
        valid = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)

        for t in range(self.n_levels):
            nb, size, off = decoded.shape[1], self.level_sizes[t], self.offsets[t]
            logits = self.t5(
                encoder_outputs=(memory.repeat_interleave(nb, 0),),
                attention_mask=mask.repeat_interleave(nb, 0),
                decoder_input_ids=decoded.view(B * nb, -1),
                use_cache=False,  # the key/value cache holds ~1 GB per layer at beam 50
            ).logits[:, -1]
            logp = logits.float().log_softmax(-1)[:, off : off + size].view(B, nb, size)

            nxt = index.next[t][prefix].long()            # [B, nb, size], -1 = no such ID
            if constrained:
                logp = logp.masked_fill(nxt < 0, -torch.inf)
            cand = (scores[:, :, None] + logp).view(B, nb * size)
            scores, flat = cand.topk(min(beam, nb * size), dim=1)
            parent, code = flat // size, flat % size

            new_prefix = nxt.view(B, nb * size).gather(1, flat)
            valid = valid.gather(1, parent) & (new_prefix >= 0) & torch.isfinite(scores)
            prefix = new_prefix.clamp_min(0)
            decoded = torch.cat(
                [decoded.gather(1, parent[:, :, None].expand(-1, -1, decoded.shape[2])),
                 (off + code)[:, :, None]], dim=2)

        items = torch.where(valid, index.leaf_item[prefix], 0)
        return items, scores
