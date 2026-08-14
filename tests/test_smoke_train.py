"""Smoke tests for tiny GPT + codec/BPE embeddings."""

from __future__ import annotations

import pytest
import torch

from embedding import BPEEmbedding, CodecEmbedding, make_embedding
from tiny_gpt import TinyGPT
from train_smoke import run_smoke


@pytest.fixture
def tiny_codec_table():
    import numpy as np

    rng = np.random.default_rng(0)
    return rng.standard_normal((128, 32)).astype(np.float32)


def test_bpe_embedding_forward():
    emb = BPEEmbedding(vocab_size=64, d_model=16)
    out = emb(torch.tensor([[1, 2, 3]]))
    assert out.shape == (1, 3, 16)


def test_codec_embedding_forward(tiny_codec_table):
    emb = CodecEmbedding(codec_table=tiny_codec_table, d_model=16, kind="kronecker")
    out = emb(torch.tensor([[1, 2, 3]]))
    assert out.shape == (1, 3, 16)
    assert emb.trainable_embedding_params == 32 * 16


def test_tiny_gpt_forward(tiny_codec_table):
    emb = CodecEmbedding(codec_table=tiny_codec_table, d_model=16, kind="fourier")
    model = TinyGPT(embedding=emb, n_layers=1, n_heads=2, max_seq_len=8)
    input_ids = torch.randint(0, 128, (2, 7))
    labels = torch.randint(0, 128, (2, 7))
    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (2, 7, 128)
    assert loss is not None
    assert torch.isfinite(loss)


@pytest.mark.parametrize("kind", ["bpe", "kronecker", "fourier", "compact"])
def test_smoke_train_100_steps(kind: str):
    result = run_smoke(
        embedding=kind,  # type: ignore[arg-type]
        steps=100,
        batch_size=4,
        seq_len=16,
        d_model=32,
        n_layers=1,
        n_heads=2,
        vocab_limit=256,
        device="cpu",
    )
    assert result["first_avg_loss"] > 0
    assert result["last_avg_loss"] > 0


def test_make_embedding_kinds():
    for kind in ("bpe", "kronecker", "fourier", "compact"):
        emb = make_embedding(kind=kind, d_model=16, vocab_limit=64, d_p=8, compact_dim=32)
        assert emb.d_model == 16
