"""Tests for real-corpus training ablation."""

from __future__ import annotations

from pathlib import Path

import pytest

from data import DEFAULT_CORPUS_SMALL, load_corpus_text, sample_batch, tokenize_corpus, train_val_split
from train import train


def test_corpus_loads_and_tokenizes():
    text = load_corpus_text()
    ids = tokenize_corpus(text, max_tokens=500)
    assert len(ids) >= 100
    train, val = train_val_split(ids, val_fraction=0.1)
    assert len(train) > len(val)


def test_english_corpus_loads():
    text = load_corpus_text()
    ids = tokenize_corpus(text)
    assert len(ids) >= 50_000


def test_indic_corpus_loads():
    from data import DEFAULT_CORPUS_INDIC

    text = load_corpus_text(DEFAULT_CORPUS_INDIC)
    ids = tokenize_corpus(text)
    assert len(ids) >= 500


def test_sample_batch_shapes():
    import numpy as np
    import torch

    ids = np.arange(200, dtype=np.int64)
    rng = np.random.default_rng(0)
    x, y = sample_batch(ids, batch_size=4, seq_len=16, device=torch.device("cpu"), rng=rng)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


@pytest.mark.parametrize("kind", ["bpe", "compact"])
def test_train_short(kind: str):
    summary = train(
        embedding=kind,  # type: ignore[arg-type]
        steps=20,
        batch_size=4,
        seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        eval_every=10,
        eval_batches=3,
        vocab_limit=512,
        corpus_path=DEFAULT_CORPUS_SMALL,
        results_dir=Path("results_test"),
    )
    assert summary["best_val_loss"] > 0
    assert Path(summary["csv_path"]).is_file()
