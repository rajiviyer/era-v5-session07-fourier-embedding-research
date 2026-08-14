"""Tokenized text corpus for tiny-LM training ablations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from byte_table import get_gpt2_tokenizer

DEFAULT_CORPUS = Path(__file__).resolve().parent / "data" / "corpus_sample.txt"
DEFAULT_CORPUS_INDIC = Path(__file__).resolve().parent / "data" / "corpus_indic_sample.txt"


def load_corpus_text(path: Path | str = DEFAULT_CORPUS) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"corpus not found: {path}")
    return path.read_text(encoding="utf-8")


def tokenize_corpus(
    text: str,
    tokenizer=None,
    max_tokens: int | None = None,
    vocab_limit: int | None = None,
) -> np.ndarray:
    """Encode UTF-8 text to GPT-2 token ids."""
    if tokenizer is None:
        tokenizer = get_gpt2_tokenizer()

    token_ids = tokenizer.encode(text)
    if vocab_limit is not None:
        token_ids = [t for t in token_ids if t < vocab_limit]
        if not token_ids:
            raise ValueError(f"no tokens remain after vocab_limit={vocab_limit}")

    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]

    return np.asarray(token_ids, dtype=np.int64)


def train_val_split(
    token_ids: np.ndarray,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    split = int(len(token_ids) * (1.0 - val_fraction))
    split = max(split, 1)
    return token_ids[:split], token_ids[split:]


def sample_batch(
    token_ids: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    """
    Sample random contiguous windows for causal LM training.

    Returns input_ids (B, seq_len) and labels (B, seq_len).
    """
    window = seq_len + 1
    ids = token_ids
    if len(ids) < window:
        reps = int(np.ceil(window / max(len(ids), 1))) + 1
        ids = np.tile(ids, reps)

    max_start = len(ids) - window
    starts = rng.integers(0, max_start + 1, size=batch_size)
    batch = np.stack([ids[s : s + window] for s in starts])
    tokens = torch.as_tensor(batch, device=device, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]


def max_seq_len_for_corpus(token_ids: np.ndarray) -> int:
    """Largest seq_len with at least one training window (needs seq_len + 1 tokens)."""
    return max(1, len(token_ids) - 2)
