"""
V2 Research #4 playground — Fourier vs Kronecker codec comparison.

    .venv\\Scripts\\python.exe playground_v2_fourier.py
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from byte_table import get_gpt2_tokenizer
from codec import codec_from_string
from fourier_codec import build_fourier_codec_matrix, fourier_codec_from_string
from probes.neighbors import (
    build_kronecker_codec_matrix,
    loose_morph_at_k,
    probe_neighbors,
    summarize_probe_families,
)
from similarity import centered_cosine


def pair_table(title: str, pairs: list[tuple[str, str]], d_p: int) -> None:
    print(f"\n  {title}")
    print(f"  {'Pair':40s}  {'Kronecker':>10s}  {'Fourier':>10s}")
    print("  " + "-" * 64)
    for a, b in pairs:
        k_cos = centered_cosine(
            codec_from_string(a, d_p=d_p),
            codec_from_string(b, d_p=d_p),
        )
        f_cos = centered_cosine(
            fourier_codec_from_string(a, d_p=d_p),
            fourier_codec_from_string(b, d_p=d_p),
        )
        print(f"  {a!r} vs {b!r}  {k_cos:10.4f}  {f_cos:10.4f}")


def experiment_1_pairwise(d_p: int = 16) -> None:
    print("=" * 70)
    print("EXPERIMENT 1: Pairwise cosine — Kronecker spikes vs Fourier waves")
    print("=" * 70)
    print("  Hypothesis: both preserve byte locality; Fourier spreads energy (dense κ).\n")

    pairs = [
        ("run", "runs"),
        ("run", "runner"),
        ("run", "Run"),
        ("run", "ran"),
        ("separate", "seperate"),
        ("swift", "SWIFT"),
        ("compute", "computer"),
        ("love", "affection"),
        ("love", "प्रेम"),
        ("compute", "commute"),
    ]
    pair_table("Typographic / typo / semantic pairs", pairs, d_p)


def experiment_2_neighbors(d_p: int = 16) -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Neighbors of 'run' — Kronecker vs Fourier")
    print("=" * 70)

    tok = get_gpt2_tokenizer()
    print(f"\n  Building codec tables (d_p={d_p})...")
    kron = build_kronecker_codec_matrix(tok, d_p=d_p)
    four = build_fourier_codec_matrix(tok, d_p=d_p)

    for name, matrix in [("Kronecker", kron), ("Fourier", four)]:
        neighbors = probe_neighbors("run", matrix, tok, k=5)
        morph = loose_morph_at_k([n for n, _ in neighbors], "run")
        print(f"\n  {name} — loose morph@5 = {morph:.2f}")
        for rank, (token_str, cos) in enumerate(neighbors, 1):
            print(f"    {rank}. {token_str!r}  cos={cos:.4f}")


def experiment_3_aggregate_morph(d_p: int = 16) -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Mean loose morph@5 across probe families")
    print("=" * 70)
    print("  Paper reference: Kronecker ~0.92, BPE ~0.54, Random ~1.0\n")

    tok = get_gpt2_tokenizer()
    kron = build_kronecker_codec_matrix(tok, d_p=d_p)
    four = build_fourier_codec_matrix(tok, d_p=d_p)

    for name, matrix in [("Kronecker", kron), ("Fourier", four)]:
        summary = summarize_probe_families(matrix, tok, name, k=5)
        agg = summary["_aggregate"]
        print(
            f"  {agg['space']:12s}  mean loose morph@5 = {agg['mean_loose_morph_at_k']:.3f}  "
            f"anisotropy = {agg['anisotropy']:.4f}"
        )


def experiment_4_wave_intuition(d_p: int = 16) -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Single-byte wave intuition (first 16 dims of raw codec)")
    print("=" * 70)

    from fourier_codec import fourier_codec

    for char in ["a", "b", "r"]:
        raw = fourier_codec([ord(char)], d_c=256, d_p=d_p)
        preview = ", ".join(f"{x:+.3f}" for x in raw[:16])
        print(f"  byte {char!r} (0x{ord(char):02x}) → first 16 dims: [{preview}, ...]")
    print("  → Each byte activates a distinct wave pattern across all D dimensions.\n")


if __name__ == "__main__":
    experiment_1_pairwise()
    experiment_2_neighbors()
    experiment_3_aggregate_morph()
    experiment_4_wave_intuition()
