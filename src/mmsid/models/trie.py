"""Prefix index over the valid semantic IDs, stored as dense transition tables.

A trie answers "which tokens may follow this prefix?". Asking that in Python for every beam of
every user is slow, so the trie is flattened: every distinct prefix of length t gets an integer,
and `next[t][prefix, code]` holds the integer of the extended prefix, or -1 if no item has it.
Looking up a whole batch of beams is then one tensor index.
"""

import numpy as np
import torch


class PrefixIndex:
    def __init__(self, ids: np.ndarray, level_sizes: list[int], device="cpu"):
        """ids: [n_items + 1, L] with row 0 as padding. Rows 1.. must be unique."""
        real = ids[1:]
        n, L = real.shape
        assert L == len(level_sizes)
        self.level_sizes = list(level_sizes)
        self.next: list[torch.Tensor] = []

        prefix = np.zeros(n, np.int64)  # every item starts at the root, prefix 0
        n_prefix = 1
        for t in range(L):
            key = prefix * level_sizes[t] + real[:, t]
            uniq, inverse = np.unique(key, return_inverse=True)
            table = np.full((n_prefix, level_sizes[t]), -1, np.int32)
            table[prefix, real[:, t]] = inverse
            self.next.append(torch.from_numpy(table).to(device))
            prefix, n_prefix = inverse, len(uniq)

        assert n_prefix == n, "semantic IDs are not unique"
        leaf_item = np.zeros(n, np.int64)
        leaf_item[prefix] = np.arange(1, n + 1)
        self.leaf_item = torch.from_numpy(leaf_item).to(device)

    def lookup(self, codes) -> int:
        """item_idx for a full code tuple, or 0 if no item has it."""
        prefix = 0
        for t, c in enumerate(codes):
            if not 0 <= c < self.level_sizes[t]:
                return 0
            prefix = int(self.next[t][prefix, c])
            if prefix < 0:
                return 0
        return int(self.leaf_item[prefix])

    def allowed(self, prefix_codes) -> list[int]:
        """Codes that may follow the given prefix."""
        prefix = 0
        for t, c in enumerate(prefix_codes):
            prefix = int(self.next[t][prefix, c])
            if prefix < 0:
                return []
        return (self.next[len(prefix_codes)][prefix] >= 0).nonzero().flatten().tolist()
