"""
Minimal causal transformer for embedding ablation smoke tests and training.

    token_ids -> input embedding -> L × (attn + FFN) -> lm_head -> logits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from embedding import count_trainable_params, make_embedding, EmbeddingKind


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Small GPT-style LM with pluggable input embedding."""

    def __init__(
        self,
        embedding: nn.Module,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = embedding
        self.vocab_size = getattr(embedding, "vocab_size", None)
        self.d_model = embedding.d_model
        self.max_seq_len = max_seq_len

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.d_model, n_heads, max_seq_len, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.d_model)
        if self.vocab_size is None:
            raise ValueError("embedding must expose vocab_size")
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    @classmethod
    def from_embedding_kind(
        cls,
        kind: EmbeddingKind,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 64,
        d_p: int = 16,
        compact_dim: int = 128,
        vocab_limit: int | None = 1024,
        **kwargs,
    ) -> TinyGPT:
        embedding = make_embedding(
            kind=kind,
            d_model=d_model,
            d_p=d_p,
            compact_dim=compact_dim,
            vocab_limit=vocab_limit,
            **kwargs,
        )
        return cls(
            embedding=embedding,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if input_ids.shape[-1] > self.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[-1]} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self.embedding(input_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
            )
        return logits, loss

    def param_summary(self) -> dict[str, int]:
        total = count_trainable_params(self)
        embed = count_trainable_params(self.embedding)
        return {
            "total_trainable": total,
            "embedding_trainable": embed,
            "transformer_trainable": total - embed,
        }


def random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Random token batch; caller slices to input[:, :-1] and labels[:, 1:]."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return tokens, tokens
