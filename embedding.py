"""
Learned input embeddings for tiny-LM ablations.

    BPE:    nn.Embedding(vocab, d_model)
    Codec:  fixed κ table (Kronecker / Fourier / Compact) + Linear(D, d_model)

Only ``proj`` (codec path) or the embedding table (BPE path) is trainable.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from byte_table import get_gpt2_tokenizer, token_id_to_bytes
from codec import kronecker_codec, truncate_utf8_bytes, z_normalize, build_kronecker_codec_matrix
from compact_codec import build_compact_codec_matrix, compact_codec
from fourier_codec import build_fourier_codec_matrix, fourier_codec

CodecKind = Literal["kronecker", "fourier", "compact"]
EmbeddingKind = Literal["bpe", "kronecker", "fourier", "compact"]


def _build_codec_matrix_np(
    kind: CodecKind,
    tokenizer,
    d_p: int,
    d_c: int,
    compact_dim: int,
    compact_combine: str,
    vocab_limit: int | None,
) -> np.ndarray:
    if vocab_limit is not None:
        vocab_size = min(vocab_limit, tokenizer.n_vocab)
        if kind == "kronecker":
            dim = d_c * d_p
            matrix = np.zeros((vocab_size, dim), dtype=np.float32)
            codec_fn = lambda seq: z_normalize(kronecker_codec(seq, d_c=d_c, d_p=d_p))
        elif kind == "fourier":
            dim = d_c * d_p
            matrix = np.zeros((vocab_size, dim), dtype=np.float32)
            codec_fn = lambda seq: z_normalize(fourier_codec(seq, d_c=d_c, d_p=d_p))
        elif kind == "compact":
            dim = compact_dim
            matrix = np.zeros((vocab_size, dim), dtype=np.float32)
            codec_fn = lambda seq: z_normalize(
                compact_codec(seq, dim=compact_dim, d_p=d_p, d_c=d_c, combine=compact_combine)
            )
        else:
            raise ValueError(f"unknown codec kind: {kind!r}")

        for token_id in range(vocab_size):
            byte_seq = truncate_utf8_bytes(token_id_to_bytes(tokenizer, token_id), d_p)
            if not byte_seq:
                continue
            matrix[token_id] = codec_fn(byte_seq)
        return matrix

    if kind == "kronecker":
        return build_kronecker_codec_matrix(tokenizer, d_p=d_p, d_c=d_c)
    if kind == "fourier":
        return build_fourier_codec_matrix(tokenizer, d_p=d_p, d_c=d_c)
    if kind == "compact":
        return build_compact_codec_matrix(
            tokenizer,
            dim=compact_dim,
            d_p=d_p,
            d_c=d_c,
            combine=compact_combine,
        )
    raise ValueError(f"unknown codec kind: {kind!r}")


class BPEEmbedding(nn.Module):
    """Standard learned token embedding (baseline)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    @property
    def codec_dim(self) -> int:
        return self.d_model

    @property
    def trainable_embedding_params(self) -> int:
        return self.vocab_size * self.d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embed(token_ids)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, d_model={self.d_model}"


class CodecEmbedding(nn.Module):
    """
    Fixed byte-local codec table + learned projection.

    Adapted from the exploratory ``kronecker_embedding.py`` module; supports
    Kronecker, Fourier, and Compact κ tables via ``build_*_codec_matrix``.
    """

    def __init__(
        self,
        codec_table: np.ndarray,
        d_model: int,
        kind: CodecKind,
        d_p: int = 16,
        d_c: int = 256,
    ):
        super().__init__()
        if codec_table.ndim != 2:
            raise ValueError(f"codec_table must be 2-D; got shape {codec_table.shape}")

        self.vocab_size, self.codec_dim = codec_table.shape
        self.d_model = d_model
        self.kind = kind
        self.d_p = d_p
        self.d_c = d_c

        self.register_buffer(
            "_codec_table",
            torch.from_numpy(np.ascontiguousarray(codec_table, dtype=np.float32)),
            persistent=False,
        )
        self.proj = nn.Linear(self.codec_dim, d_model, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(self.codec_dim))

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        d_model: int,
        kind: CodecKind = "kronecker",
        d_p: int = 16,
        d_c: int = 256,
        compact_dim: int = 128,
        compact_combine: str = "bind",
        vocab_limit: int | None = None,
    ) -> CodecEmbedding:
        matrix = _build_codec_matrix_np(
            kind=kind,
            tokenizer=tokenizer,
            d_p=d_p,
            d_c=d_c,
            compact_dim=compact_dim,
            compact_combine=compact_combine,
            vocab_limit=vocab_limit,
        )
        return cls(
            codec_table=matrix,
            d_model=d_model,
            kind=kind,
            d_p=d_p,
            d_c=d_c,
        )

    @property
    def trainable_embedding_params(self) -> int:
        return self.codec_dim * self.d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        flat_ids = token_ids.reshape(-1)
        codec_out = self._codec_table.index_select(0, flat_ids)
        codec_out = codec_out.view(*token_ids.shape, self.codec_dim)
        return self.proj(codec_out.to(self.proj.weight.dtype))

    def extra_repr(self) -> str:
        return (
            f"kind={self.kind!r}, vocab_size={self.vocab_size}, "
            f"d_model={self.d_model}, codec_dim={self.codec_dim}, "
            f"d_c={self.d_c}, d_p={self.d_p}"
        )


def make_embedding(
    kind: EmbeddingKind,
    d_model: int,
    tokenizer=None,
    d_p: int = 16,
    d_c: int = 256,
    compact_dim: int = 128,
    compact_combine: str = "bind",
    vocab_limit: int | None = None,
) -> nn.Module:
    """
    Factory for BPE or codec input embeddings.

    Args:
        kind: ``bpe``, ``kronecker``, ``fourier``, or ``compact``.
        vocab_limit: If set, use only token ids ``0 .. vocab_limit-1`` (fast smoke tests).
    """
    if tokenizer is None:
        tokenizer = get_gpt2_tokenizer()

    if kind == "bpe":
        vocab_size = vocab_limit if vocab_limit is not None else tokenizer.n_vocab
        return BPEEmbedding(vocab_size=vocab_size, d_model=d_model)

    if kind not in ("kronecker", "fourier", "compact"):
        raise ValueError(f"unknown embedding kind: {kind!r}")

    return CodecEmbedding.from_tokenizer(
        tokenizer=tokenizer,
        d_model=d_model,
        kind=kind,
        d_p=d_p,
        d_c=d_c,
        compact_dim=compact_dim,
        compact_combine=compact_combine,
        vocab_limit=vocab_limit,
    )


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
