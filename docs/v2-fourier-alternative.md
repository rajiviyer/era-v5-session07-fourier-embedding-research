# V2 Research #4: Fourier Alternative to Kronecker

**Problem:** What is a real Fourier alternative to Kronecker? Can each character be a Fourier wave, summed to form a word?

**Solution:** Replace Kronecker **spikes** with **sinusoidal basis waves**; same `D = d_c × d_p`, same z-norm, same `proj` interface.

---

## Kronecker vs Fourier

| | Kronecker (V1) | Fourier (V2) |
|---|----------------|--------------|
| Per (byte, position) | Spike at index `byte × d_p + pos` | Wave `cos + sin` across all `D` dims |
| Raw κ sparsity | Sparse (L nonzero) | Dense (all D active) |
| Superposition | Add spikes | Add waves (Fourier synthesis) |
| Output dim | `D = 256 × d_p` | Same `D` |
| z-norm + `proj` | Unchanged | Unchanged |

### Formula

For byte sequence `b₁…b_L` (truncated to `d_p`), linear index `i_p = b_p × d_p + p`:

**Kronecker:**
```
κ_K[i_p] += 1/√L
```

**Fourier:**
```
κ_F(j) += (1/√L) · (cos(2π j i_p / D) + sin(2π j i_p / D)) / √2
```

Word embedding = sum of per-byte waves → classic superposition.

---

## How we prove it works

Two layers (see root [README.md](../README.md)):

**Layer 1 — Geometry (no LM):** Phase 4-style probes on the fixed codec:

1. **Typo robustness:** `separate` / `seperate` cosine stays high.
2. **Case sensitivity:** `run` / `Run` / `swift` / `SWIFT` stay distinct.
3. **Prefix locality:** `run` / `runs` / `runner` cluster.
4. **No false semantics:** `love` / `affection`, `love` / `प्रेम` stay low.
5. **NN probe:** loose morph@5 on `"run"` family; compare to Kronecker.
6. **Distinct from Kronecker:** same string → correlated but not identical vectors.

**Layer 2 — Tiny LM:** Train identical TinyGPT on real text with BPE vs Kronecker vs Fourier vs Compact; val loss must decrease under the same corpus and hyperparameters.

```bash
uv run python main.py
uv run python train.py --all --steps 500 --eval-every 100 --plot
```

---

## Code

| File | Purpose |
|------|---------|
| `fourier_codec.py` | `fourier_codec`, `fourier_codec_from_string`, `build_fourier_codec_matrix` |
| `playground_v2_fourier.py` | Side-by-side Kronecker vs Fourier vs Compact experiments |
| `embedding.py` | PyTorch `CodecEmbedding` with Fourier table + learned `proj` |
| `train.py` | Real-corpus 4-way LM ablation |
| `tests/test_fourier_codec.py` | Unit tests |

---

## Run

```bash
uv sync --group dev
uv run pytest tests/test_fourier_codec.py -v
uv run python playground_v2_fourier.py
```

Building full vocab Fourier table (`d_p=16`) takes ~30–60s on first run.

---

## Measured findings (`playground_v2_fourier.py`, `d_p=16`)

| Phenomenon | Kronecker | Fourier |
|------------|-----------|---------|
| Typos (`separate`/`seperate`) | 0.875 | **0.875** |
| Case (`swift`/`SWIFT`) | −0.001 | **0.000** |
| Prefix (`run`/`runs`) | 0.866 | **0.866** |
| False semantics (`love`/`affection`) | 0.166 | **0.167** |
| NN of `run` | `runs`, `ru`, `runner` | **Identical top-5** |
| Mean loose morph@5 | 0.940 | **0.940** |
| Same string vs Kronecker vector | · | cosine ≈ **0.002** |

**Insight:** Fourier waves keyed by the same `lin_idx = byte × d_p + pos` preserve **byte-locality geometry** almost exactly after z-norm, while producing **dense** vectors. Probe rankings coincide; raw κ subspaces differ.

**Tradeoff vs Kronecker spikes:** Dense κ activates all D coordinates each forward pass; collision analysis and matched-parameter LM ablations are future work (see README).

---

## Limitations & next steps

- Fixed `d_p` truncation (dynamic positions are separate research).
- Harmonics / learned frequencies could extend the basis.
- Invertibility not addressed; dense κ may have more collisions than sparse spikes.
- See [v3-compact-codec.md](v3-compact-codec.md) for low-D factored alternative (Compact bind).
