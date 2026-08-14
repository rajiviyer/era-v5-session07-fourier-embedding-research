"""
V2 Research #4 playground — Fourier vs Kronecker codec comparison.

    .venv\\Scripts\\python.exe playground_v2_fourier.py
"""

from __future__ import annotations

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from byte_table import get_gpt2_tokenizer
from codec import codec_from_string
from compact_codec import build_compact_codec_matrix, compact_codec_from_string
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


PROBE_PAIRS = [
    ("separate", "seperate"),
    ("swift", "SWIFT"),
    ("run", "runs"),
    ("love", "affection"),
]


def _pairwise_score(pairs: list[tuple[str, str]], codec_fn) -> float:
    scores = [centered_cosine(codec_fn(a), codec_fn(b)) for a, b in pairs]
    return float(sum(scores) / len(scores))


def experiment_5_compact_sweep(d_p: int = 16) -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Compact factored codec — dimension sweep (D ≪ 256×d_p)")
    print("=" * 70)
    print(
        "  Hypothesis: byte_wave(b) ⊙ position_wave(p) + superposition preserves "
        "geometry at small D.\n"
    )

    kronecker_dim = 256 * d_p
    print(f"  Kronecker / full Fourier baseline D = {kronecker_dim}\n")

    def kron_fn(s: str) -> np.ndarray:
        return codec_from_string(s, d_p=d_p)

    kron_morph = _pairwise_score(PROBE_PAIRS, kron_fn)
    print(f"  {'Codec':28s}  {'D':>6s}  {'mean pair cos':>14s}")
    print("  " + "-" * 54)
    print(f"  {'Kronecker (baseline)':28s}  {kronecker_dim:6d}  {kron_morph:14.4f}")

    for dim in (64, 128, 256, 512, 1024):
        def compact_fn(s: str, d: int = dim) -> np.ndarray:
            return compact_codec_from_string(s, dim=d, d_p=d_p, combine="bind")

        score = _pairwise_score(PROBE_PAIRS, compact_fn)
        ratio = dim / kronecker_dim
        print(f"  {'Compact bind':28s}  {dim:6d}  {score:14.4f}  ({ratio:.1%} of baseline D)")

    tok = get_gpt2_tokenizer()
    print(f"\n  Full-vocab morph@5 (d_p={d_p})...")
    kron_matrix = build_kronecker_codec_matrix(tok, d_p=d_p)
    kron_summary = summarize_probe_families(kron_matrix, tok, "Kronecker", k=5)
    kron_m = kron_summary["_aggregate"]["mean_loose_morph_at_k"]

    for dim in (128, 256, 512):
        matrix = build_compact_codec_matrix(tok, dim=dim, d_p=d_p, combine="bind")
        summary = summarize_probe_families(matrix, tok, f"Compact@{dim}", k=5)
        morph = summary["_aggregate"]["mean_loose_morph_at_k"]
        print(
            f"  Compact bind D={dim:4d}  morph@5={morph:.3f}  "
            f"(Kronecker={kron_m:.3f}, {100 * dim / kronecker_dim:.1f}% dims)"
        )
    print()


if __name__ == "__main__":
    experiment_1_pairwise()
    experiment_2_neighbors()
    experiment_3_aggregate_morph()
    experiment_4_wave_intuition()
    experiment_5_compact_sweep()
