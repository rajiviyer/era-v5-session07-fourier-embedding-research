"""
Phase 2: Connect BPE token IDs to UTF-8 byte sequences (Paper §3.2).

Pipeline:
    token_id -> surface-form UTF-8 bytes (from tokenizer) -> κ(b)

tiktoken: decode_single_token_bytes (byte-BPE fragments are often invalid UTF-8 alone).
HuggingFace: convert_ids_to_tokens / decode path (see token_id_to_bytes).
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from codec import truncate_utf8_bytes

# SentencePiece-style byte fallback: "<0xC3>" represents single byte 0xC3
_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9a-fA-F]{2})>$")


class Tokenizer(Protocol):
    """Minimal interface for tiktoken or HuggingFace tokenizers."""

    def decode(self, token_ids: list[int]) -> str: ...


def get_gpt2_tokenizer():
    """Return tiktoken GPT-2 encoding (~50,257 vocab)."""
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def token_string_to_bytes(token_str: str) -> list[int]:
    """
    Convert a vocab entry's surface string to UTF-8 bytes.

    Handles SentencePiece byte-fallback tokens like "<0xC3>" -> [0xC3].
    All other strings are UTF-8 encoded literally (special tokens included).
    """
    match = _BYTE_FALLBACK_RE.match(token_str)
    if match:
        return [int(match.group(1), 16)]
    return list(token_str.encode("utf-8"))


def token_id_to_bytes(tokenizer: Tokenizer, token_id: int) -> list[int]:
    """
    Return full UTF-8 byte values (0-255) for this vocab entry (before d_p truncation).

    Args:
        tokenizer: tiktoken encoding or HuggingFace tokenizer.
        token_id: Integer token id in [0, vocab_size).
    """
    decode_single = getattr(tokenizer, "decode_single_token_bytes", None)
    if decode_single is not None:
        return list(decode_single(token_id))

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece:
            match = _BYTE_FALLBACK_RE.match(piece)
            if match:
                return [int(match.group(1), 16)]
        if token_id in special_ids:
            return list((piece or "").encode("utf-8"))
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return list((decoded or "").encode("utf-8"))

    token_str = tokenizer.decode([token_id])
    return token_string_to_bytes(token_str)


def build_byte_tables(
    tokenizer: Tokenizer,
    d_p: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute byte buffer for the full vocabulary.

    Returns:
        byte_table:   shape [vocab_size, d_p], uint8, zero-padded, UTF-8-safe truncated
        length_table: shape [vocab_size], int16, actual byte length stored per token
    """
    vocab_size = tokenizer.n_vocab
    byte_table = np.zeros((vocab_size, d_p), dtype=np.uint8)
    length_table = np.zeros(vocab_size, dtype=np.int16)
    for token_id in range(vocab_size):
        bs = truncate_utf8_bytes(token_id_to_bytes(tokenizer, token_id), d_p)
        length_table[token_id] = len(bs)
        byte_table[token_id, :len(bs)] = bs
        
    return byte_table, length_table


def compute_truncation_stats(tokenizer, d_p: int) -> dict[str, float | int]:
    """
    Scan full vocab: how many token strings exceed d_p UTF-8 bytes before truncation?
    """
    vocab_size = tokenizer.n_vocab  # tiktoken
    needs_trunc = 0
    for token_id in range(vocab_size):
        raw_len = len(token_id_to_bytes(tokenizer, token_id))
        if raw_len > d_p:
            needs_trunc += 1
    return {
        "vocab_size": vocab_size,
        "d_p": d_p,
        "needs_truncation": needs_trunc,
        "fraction_truncated": needs_trunc / vocab_size,
        "fraction_covered": 1.0 - needs_trunc / vocab_size,
    }
