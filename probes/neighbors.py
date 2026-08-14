"""
Phase 4: Nearest-neighbor geometry probes (Paper §6.1–6.4).

Compare embedding spaces via mean-centered cosine similarity:
  trained BPE | random Gaussian | Kronecker codec κ
"""

from __future__ import annotations

from typing import Any

import numpy as np

from codec import build_kronecker_codec_matrix

# Paper §6.1 probe families
PROBE_FAMILIES: dict[str, list[str]] = {
    "run": ["run", "runs", "running", "runner", "ran"],
    "compute": ["compute", "computer", "computing", "computation", "computes"],
    "magnet": ["magnet", "magnets", "magnetic", "magnetize", "magnetized"],
    "tion": ["nation", "station", "action", "rotation", "creation"],
}


def mean_center(matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """Subtract column mean across vocabulary rows (Paper §6.1)."""
    return matrix - matrix.mean(axis=axis, keepdims=True)


def anisotropy_norm(matrix: np.ndarray) -> float:
    """L2 norm of the vocabulary mean vector (Paper §6.4)."""
    return float(np.linalg.norm(matrix.mean(axis=0)))


def canonical_form(s: str) -> str:
    """Normalize surface form for loose morph@K (Paper §6.1)."""
    s = s.replace("\u2581", " ").replace("\u0120", " ")
    s = s.strip(" \t\n\r.,;:!?\"''()[]{}_-/\\<>")
    return s.casefold()


def loose_morph_at_k(neighbors: list[str], probe: str) -> float:
    """
    Fraction of neighbors whose canonical form differs from the probe's.

    Typographic variants (run → Run) score 0; byte-different strings score 1.
    """
    if not neighbors:
        return 0.0
    probe_canon = canonical_form(probe)
    escapes = sum(1 for n in neighbors if canonical_form(n) != probe_canon)
    return escapes / len(neighbors)


def random_gaussian_matrix(
    vocab_size: int,
    dim: int,
    seed: int = 0,
) -> np.ndarray:
    """Untrained embedding baseline (Paper §6.1)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((vocab_size, dim)).astype(np.float32)


def query_vector_for_probe(
    probe: str,
    embedding_matrix: np.ndarray,
    tokenizer,
) -> tuple[np.ndarray, list[int]]:
    """
    Build query vector for a probe string (Paper §6.1).

    Single BPE token → that row; multi-token → mean of subtoken rows.
    """
    token_ids = tokenizer.encode(probe)
    if not token_ids:
        raise ValueError(f"probe {probe!r} tokenized to empty id list")
    rows = embedding_matrix[np.array(token_ids, dtype=np.int64)]
    if len(token_ids) == 1:
        return rows[0], token_ids
    return rows.mean(axis=0), token_ids


def decode_token(tokenizer, token_id: int) -> str:
    """Best-effort decode of one vocab id for display."""
    return tokenizer.decode([token_id])


def nearest_neighbors(
    query_vec: np.ndarray,
    embedding_matrix: np.ndarray,
    tokenizer,
    exclude_token_ids: set[int] | list[int] | None = None,
    k: int = 5,
) -> list[tuple[str, float]]:
    """
    Top-k mean-centered cosine neighbors, excluding given token ids.

    Returns:
        List of (decoded_token_string, cosine_similarity).
    """
    if exclude_token_ids is None:
        exclude_ids: set[int] = set()
    else:
        exclude_ids = set(exclude_token_ids)

    centered = mean_center(embedding_matrix)
    query_centered = query_vec - embedding_matrix.mean(axis=0)

    q_norm = np.linalg.norm(query_centered)
    if q_norm < 1e-12:
        return []

    row_norms = np.linalg.norm(centered, axis=1)
    dots = centered @ query_centered
    denom = row_norms * q_norm + 1e-8
    cosines = dots / denom

    for tid in exclude_ids:
        if 0 <= tid < len(cosines):
            cosines[tid] = -np.inf

    top_idx = np.argsort(cosines)[::-1][:k]
    results: list[tuple[str, float]] = []
    for idx in top_idx:
        if cosines[idx] == -np.inf:
            continue
        results.append((decode_token(tokenizer, int(idx)), float(cosines[idx])))
    return results


def probe_neighbors(
    probe: str,
    embedding_matrix: np.ndarray,
    tokenizer,
    k: int = 5,
) -> list[tuple[str, float]]:
    """End-to-end NN probe for one string."""
    query_vec, token_ids = query_vector_for_probe(probe, embedding_matrix, tokenizer)
    return nearest_neighbors(
        query_vec,
        embedding_matrix,
        tokenizer,
        exclude_token_ids=set(token_ids),
        k=k,
    )


def load_hf_input_embeddings(model_id: str) -> tuple[np.ndarray, Any]:
    """
    Load pretrained input embedding table from HuggingFace.

    Requires: transformers, safetensors
    Auth: set HF_TOKEN in environment or `.env` (see HuggingFace hub docs).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required for pretrained BPE probes. "
            "Install with: uv add --dev transformers safetensors"
        ) from e

    from hf_env import configure_hf_hub

    token = configure_hf_hub()
    kwargs: dict[str, Any] = {"dtype": torch.float32}
    if token:
        kwargs["token"] = token

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    embed = model.get_input_embeddings()
    # SmolLM2 stores embeddings in bfloat16; NumPy needs float32.
    matrix = embed.weight.detach().cpu().to(dtype=torch.float32).numpy()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    return matrix, tokenizer


def summarize_probe_families(
    embedding_matrix: np.ndarray,
    tokenizer,
    space_name: str,
    k: int = 5,
) -> dict:
    """Run all paper probe families; return neighbors + loose morph@K per probe."""
    report: dict = {"families": {}, "_aggregate": {}}
    morph_scores: list[float] = []
    for family, probes in PROBE_FAMILIES.items():
        report["families"][family] = {}
        for probe in probes:
            neighbors = probe_neighbors(probe, embedding_matrix, tokenizer, k=k)
            neighbor_strs = [n for n, _ in neighbors]
            morph = loose_morph_at_k(neighbor_strs, probe)
            morph_scores.append(morph)
            report["families"][family][probe] = {
                "neighbors": neighbors,
                "loose_morph_at_k": morph,
            }
    report["_aggregate"] = {
        "space": space_name,
        "mean_loose_morph_at_k": float(np.mean(morph_scores)),
        "anisotropy": anisotropy_norm(embedding_matrix),
    }
    return report
