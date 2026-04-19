"""
Neural Engram module — follows the architecture in Cheng et al. (2025)
"Conditional Memory via Scalable Lookup".

Components:
  1. Multi-head hashing: suffix n-grams → embedding table indices
  2. Context-aware scalar gate: α = σ(RMSNorm(h)·RMSNorm(W_K·e) / √d)
  3. Gated value:  ṽ = α · (W_V · e)
  4. Depthwise causal convolution + SiLU + skip connection
  5. Residual add back into backbone hidden states

Trust integration (paper §6.3 analogue):
  The optional `hygiene_mask` scales α per position, suppressing the
  Engram contribution for n-gram slots that are polluted by bad training
  data — bridging the static trust scores in engram_trust.py with the
  learned gate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms


def hash_ngrams(
    canonical_ids: torch.Tensor,  # [B, T] long
    n: int,
    head: int,
    num_buckets: int,
) -> torch.Tensor:  # [B, T] long; positions 0..n-2 are bucket 0 (no n-gram yet)
    """Vectorised polynomial hash for all sliding n-gram windows.

    Uses torch.unfold for O(T) slicing and a per-order, per-head
    seed so different (n, head) pairs map to different buckets.
    """
    B, T = canonical_ids.shape
    if T < n:
        return torch.zeros(B, T, dtype=torch.long, device=canonical_ids.device)

    # windows: [B, T-n+1, n]
    windows = canonical_ids.unfold(dimension=1, size=n, step=1)

    primes = torch.tensor(
        [7919, 104723, 224737, 350377, 479909, 611957][:n],
        dtype=torch.long,
        device=canonical_ids.device,
    )
    # Polynomial hash: sum of token * prime[position]
    h = (windows * primes).sum(dim=-1)            # [B, T-n+1]
    h = (h + head * 31337) % num_buckets          # head-specific offset

    # Positions 0..n-2 have no complete n-gram; pad with 0
    pad = torch.zeros(B, n - 1, dtype=torch.long, device=canonical_ids.device)
    return torch.cat([pad, h.to(torch.long)], dim=1)   # [B, T]


class EngramModule(nn.Module):
    """
    Plug-in neural memory module for transformer backbones.

    Args:
        d_model:          Hidden dimension of the backbone.
        max_ngram:        Highest n-gram order (paper uses 3, i.e. bi- and tri-grams).
        num_hash_heads:   Hash heads per n-gram order (paper uses 8; 4 is a good default).
        num_buckets:      Embedding table size per (order, head). Use a prime number.
                          With 2 orders × 4 heads × 16381 × 128 d_head ≈ 67 M params.
        conv_kernel_size: Depthwise causal conv kernel (paper uses 4).
    """

    def __init__(
        self,
        d_model: int,
        max_ngram: int = 3,
        num_hash_heads: int = 4,
        num_buckets: int = 16381,
        conv_kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_ngram = max_ngram
        self.num_hash_heads = num_hash_heads
        self.num_buckets = num_buckets
        self.conv_kernel_size = conv_kernel_size

        self.ngram_orders = list(range(2, max_ngram + 1))  # [2, 3, ...]
        num_tables = len(self.ngram_orders) * num_hash_heads

        if d_model % num_tables != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_tables={num_tables} "
                f"(ngram_orders={self.ngram_orders}, num_hash_heads={num_hash_heads})"
            )
        self.d_head = d_model // num_tables  # per-table embedding dimension

        # One embedding table per (n-gram order, hash head)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_buckets, self.d_head) for _ in range(num_tables)]
        )

        # Attention gate: project memory to key and value in d_model space
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        self.norm_q = RMSNorm(d_model)  # normalise backbone hidden state (query)
        self.norm_k = RMSNorm(d_model)  # normalise projected key
        self.norm_v = RMSNorm(d_model)  # normalise gated value before conv

        # Depthwise causal convolution (paper eq. 5, kernel_size=4, dilation=max_ngram)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel_size,
            groups=d_model,               # depthwise
            padding=conv_kernel_size - 1, # left-pad then truncate → causal
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embeddings:
            nn.init.zeros_(emb.weight)
        # Zero-init conv → identity at start; gates drive learning from scratch
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def retrieve(self, canonical_ids: torch.Tensor) -> torch.Tensor:
        """Look up and concatenate all n-gram embeddings.

        Returns: [B, T, d_model]
        """
        parts = []
        idx = 0
        for n in self.ngram_orders:
            for head in range(self.num_hash_heads):
                buckets = hash_ngrams(canonical_ids, n, head, self.num_buckets)
                parts.append(self.embeddings[idx](buckets))  # [B, T, d_head]
                idx += 1
        return torch.cat(parts, dim=-1)  # [B, T, d_model]

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, T, d_model]
        canonical_ids: torch.Tensor,           # [B, T] long
        hygiene_mask: Optional[torch.Tensor] = None,  # [B, T] float ∈ [0,1]
    ) -> torch.Tensor:
        """
        Returns hidden_states + Y (residual).

        hygiene_mask: per-position trust hygiene (1 = fully trusted, 0 = polluted).
        Scales the attention gate α, suppressing low-hygiene n-gram slots —
        the neural analogue of engram_trust.py's bad_hits / quarantine logic.
        """
        B, T, d = hidden_states.shape

        # 1. Retrieve static memory vector from embedding tables
        e = self.retrieve(canonical_ids)          # [B, T, d_model]

        # 2. Context-aware gating (paper §2.3)
        k = self.W_K(e)
        v = self.W_V(e)

        alpha = torch.sigmoid(
            (self.norm_q(hidden_states) * self.norm_k(k)).sum(dim=-1, keepdim=True)
            / math.sqrt(d)
        )  # [B, T, 1]

        # Optionally scale gate by hygiene score from the symbolic trust layer
        if hygiene_mask is not None:
            alpha = alpha * hygiene_mask.unsqueeze(-1)

        v_tilde = alpha * v  # [B, T, d_model]

        # 3. Depthwise causal conv + SiLU skip (paper eq. 5)
        x = self.norm_v(v_tilde).transpose(1, 2)   # [B, d_model, T]
        x = self.conv(x)[..., :T]                   # causal: keep only first T outputs
        Y = F.silu(x.transpose(1, 2)) + v_tilde     # [B, T, d_model]

        return hidden_states + Y
