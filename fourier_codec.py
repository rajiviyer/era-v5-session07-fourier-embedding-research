"""
V2 Research #4: Fourier alternative to the Kronecker spike codec.

Kronecker places a single spike at lin_idx = byte × d_p + position.

Fourier sums a sinusoidal basis vector for each (byte, position) pair:

    κ_F(j) = (1/√L) Σ_p  cos(2π j · lin_idx_p / D) + sin(2π j · lin_idx_p / D)
             ────────────────────────────────────────────────────────────────
                                    √2 per term

Each character contributes a wave; a word is the superposition of its byte waves.
Same output dimension D = d_c × d_p as Kronecker → drop-in for W_proj.
"""

from __future__ import annotations

import numpy as np

from codec import truncate_utf8_bytes, z_normalize


def _basis_wave(lin_idx: int, d_c: int, d_p: int) -> np.ndarray:
    """Unit-norm cos+sin wave keyed by linear index lin_idx (shape D)."""
    d = d_c * d_p
    j = np.arange(d, dtype=np.float64)
    phase = 2.0 * np.pi * j * lin_idx / d
    wave = (np.cos(phase) + np.sin(phase)) / np.sqrt(2.0)
    return wave


def fourier_codec(
    byte_sequence: list[int],
    d_c: int = 256,
    d_p: int = 32,
) -> np.ndarray:
    """
    Raw Fourier codec vector (no z-normalization).

    Args:
        byte_sequence: UTF-8 byte values in 0..255.
        d_c: Byte alphabet size (256).
        d_p: Max byte positions per token.

    Returns:
        Vector of shape (d_c * d_p,) — dense superposition of byte waves.
    """
    d = d_c * d_p
    kappa = np.zeros(d, dtype=np.float64)

    truncated = truncate_utf8_bytes(byte_sequence, d_p)
    length = len(truncated)
    if length == 0:
        return kappa

    scale = 1.0 / np.sqrt(length)
    for position, byte_value in enumerate(truncated):
        lin_idx = byte_value * d_p + position
        kappa += scale * _basis_wave(lin_idx, d_c, d_p)

    return kappa


def fourier_codec_from_string(s: str, d_c: int = 256, d_p: int = 32) -> np.ndarray:
    """UTF-8 string → fourier_codec → z_normalize."""
    return z_normalize(fourier_codec(list(s.encode("utf-8")), d_c=d_c, d_p=d_p))


def build_fourier_codec_matrix(
    tokenizer,
    d_p: int = 32,
    d_c: int = 256,
) -> np.ndarray:
    """Full [vocab, D] z-normalized Fourier table for NN probes."""
    from byte_table import build_byte_tables

    byte_table, length_table = build_byte_tables(tokenizer, d_p=d_p)
    vocab_size = byte_table.shape[0]
    d = d_c * d_p
    matrix = np.zeros((vocab_size, d), dtype=np.float32)
    for token_id in range(vocab_size):
        nbytes = int(length_table[token_id])
        if nbytes == 0:
            continue
        byte_seq = byte_table[token_id, :nbytes].tolist()
        matrix[token_id] = z_normalize(fourier_codec(byte_seq, d_c=d_c, d_p=d_p))
    return matrix
