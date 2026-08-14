"""
V3: Compact byte-local codec — factored byte + sinusoidal position features.

Instead of Kronecker's full cross product (D = d_c × d_p), each byte at position p
contributes a D-dimensional wave built from two factored parts:

    φ(b, p) = combine( byte_wave(b), position_wave(p) )

Token codec is the usual superposition over bytes:

    κ(b) = (1/√L) Σ_p φ(b_p, p)

combine ∈ {"add", "bind"}:
  - add:  byte_wave + position_wave  (cheap, interpretable)
  - bind: byte_wave * position_wave (Hadamard — closer to Kronecker coupling)

Default D = 256 ≪ 4096 (Kronecker at d_p=16) or 8192 (d_p=32).
"""

from __future__ import annotations

import numpy as np

from codec import truncate_utf8_bytes, z_normalize


def _sin_cos_wave(value: float, dim: int, scale: float) -> np.ndarray:
    """Unit-norm cos+sin wave for scalar value (byte index or position)."""
    j = np.arange(dim, dtype=np.float64)
    phase = 2.0 * np.pi * j * value / scale
    return (np.cos(phase) + np.sin(phase)) / np.sqrt(2.0)


def _byte_wave(byte_value: int, dim: int, d_c: int = 256) -> np.ndarray:
    return _sin_cos_wave(float(byte_value), dim, float(d_c))


def _position_wave(position: int, dim: int, d_p: int) -> np.ndarray:
    return _sin_cos_wave(float(position), dim, float(max(d_p, 1)))


def _combine(byte_part: np.ndarray, pos_part: np.ndarray, mode: str) -> np.ndarray:
    if mode == "add":
        return byte_part + pos_part
    if mode == "bind":
        return byte_part * pos_part
    raise ValueError(f"unknown combine mode: {mode!r}")


def compact_codec(
    byte_sequence: list[int],
    dim: int = 256,
    d_p: int = 32,
    d_c: int = 256,
    combine: str = "bind",
) -> np.ndarray:
    """
    Raw compact codec vector (no z-normalization).

    Args:
        byte_sequence: UTF-8 byte values in 0..255.
        dim: Output dimension D (typically 128–512).
        d_p: Max byte positions per token.
        d_c: Byte alphabet size for phase scaling.
        combine: "add" or "bind" (Hadamard).

    Returns:
        Vector of shape (dim,).
    """
    kappa = np.zeros(dim, dtype=np.float64)
    truncated = truncate_utf8_bytes(byte_sequence, d_p)
    length = len(truncated)
    if length == 0:
        return kappa

    scale = 1.0 / np.sqrt(length)
    for position, byte_value in enumerate(truncated):
        byte_part = _byte_wave(byte_value, dim, d_c=d_c)
        pos_part = _position_wave(position, dim, d_p=d_p)
        kappa += scale * _combine(byte_part, pos_part, combine)

    return kappa


def compact_codec_from_string(
    s: str,
    dim: int = 256,
    d_p: int = 32,
    d_c: int = 256,
    combine: str = "bind",
) -> np.ndarray:
    """UTF-8 string → compact_codec → z_normalize."""
    return z_normalize(
        compact_codec(list(s.encode("utf-8")), dim=dim, d_p=d_p, d_c=d_c, combine=combine)
    )


def build_compact_codec_matrix(
    tokenizer,
    dim: int = 256,
    d_p: int = 32,
    d_c: int = 256,
    combine: str = "bind",
) -> np.ndarray:
    """Full [vocab, D] z-normalized compact codec table for NN probes."""
    from byte_table import build_byte_tables

    byte_table, length_table = build_byte_tables(tokenizer, d_p=d_p)
    vocab_size = byte_table.shape[0]
    matrix = np.zeros((vocab_size, dim), dtype=np.float32)
    for token_id in range(vocab_size):
        nbytes = int(length_table[token_id])
        if nbytes == 0:
            continue
        byte_seq = byte_table[token_id, :nbytes].tolist()
        matrix[token_id] = z_normalize(
            compact_codec(byte_seq, dim=dim, d_p=d_p, d_c=d_c, combine=combine)
        )
    return matrix
