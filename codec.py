"""
Phase 1: Kronecker codec (Paper §3.2–3.3).

κ(b) = (1/√L) Σ_p  c_{b_p} ⊗ p_p

Each (byte, position) pair contributes a single 1 at linear index:
    lin_idx = byte_value * d_p + position
"""

from __future__ import annotations

import numpy as np


def truncate_utf8_bytes(byte_sequence: list[int], d_p: int) -> list[int]:
    """
    Truncate to at most d_p bytes, backing off to a UTF-8 codepoint boundary.

    If d_p falls inside a multibyte character, drop bytes from the end until
    the sequence is valid UTF-8 (Paper §3.2).
    """
    if len(byte_sequence) <= d_p:
        return byte_sequence

    truncated = byte_sequence[:d_p]
    while truncated:
        try:
            bytes(truncated).decode("utf-8")
            return truncated
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return []


def kronecker_codec(
    byte_sequence: list[int],
    d_c: int = 256,
    d_p: int = 32,
) -> np.ndarray:
    """
    Compute raw codec vector κ(b) without z-normalization.

    Args:
        byte_sequence: UTF-8 byte values in 0..255.
        d_c: Byte alphabet size (256).
        d_p: Max byte positions per token.

    Returns:
        Vector of shape (d_c * d_p,). At most L nonzero entries, each 1/√L.
    """
    # D = default 256 x 32 = 8192 (d_c * d_p)
    D = d_c * d_p

    # kappa = vector of shape (D,) initialized to all zeros
    kappa = np.zeros(D, dtype=np.float64)

    # truncate the byte sequence to at most d_p bytes, 
    # ensuring valid UTF-8, but dropping bytes from the end if necessary
    truncated = truncate_utf8_bytes(byte_sequence, d_p)
    L = len(truncated)
    if L == 0:
        return kappa

    scale = 1.0 / np.sqrt(L)
    for position, byte_value in enumerate(truncated):
        lin_idx = byte_value * d_p + position
        kappa[lin_idx] += scale

    return kappa


def z_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-token z-normalization across D coordinates (Paper §3.3).

    Rescale v to mean 0 and std 1. If std is near zero, return zeros.
    """
    mean = v.mean()
    std = v.std()
    if std <= eps:
        return np.zeros_like(v)
    return (v - mean) / std


def codec_from_string(s: str, d_c: int = 256, d_p: int = 32) -> np.ndarray:
    """UTF-8 encode string → kronecker_codec → z_normalize.
    The idea is to make the codec vector dense with zeros becoming negative values
    and positive values becoming larger. Eventhough the vector started sparse
    """
    byte_sequence = list(s.encode("utf-8"))
    return z_normalize(kronecker_codec(byte_sequence, d_c, d_p))
