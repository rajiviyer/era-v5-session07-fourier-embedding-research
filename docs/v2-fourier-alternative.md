# V2 Research #4: Fourier Alternative to Kronecker

**Problem (from [RESEARCH_POINTS.md](../RESEARCH_POINTS.md)):**  
What is a real Fourier alternative to Kronecker? Can each character be a Fourier wave, summed to form a word?

**Solution:** Replace Kronecker **spikes** with **sinusoidal basis waves**; same `D = d_c × d_p`, same z-norm, same `W_proj` interface.

---

## Kronecker vs Fourier

| | Kronecker (V1) | Fourier (V2 proposal) |
|---|----------------|----------------------|
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

## How we prove it works (without full LM training)

Phase 4-style **geometry probes** on the codec alone:

1. **Typo robustness:** `separate` / `seperate` cosine stays high (byte similarity preserved).
2. **Case sensitivity:** `run` / `Run` / `swift` / `SWIFT` stay distinct (different bytes → different frequencies).
3. **Prefix locality:** `run` / `runs` / `runner` cluster.
4. **No false semantics:** `love` / `affection`, `love` / `प्रेम` stay low.
5. **NN probe:** loose morph@5 on `"run"` family; compare to Kronecker ~0.92.
6. **Distinct from Kronecker:** same string → correlated but not identical vectors.

Training (Phase 5) is **optional** for this submission; geometry proof matches RESEARCH_POINTS guidance.

---

## Code

| File | Purpose |
|------|---------|
| `kronecker-learn/fourier_codec.py` | `fourier_codec`, `fourier_codec_from_string`, `build_fourier_codec_matrix` |
| `kronecker-learn/playground_v2_fourier.py` | Side-by-side Kronecker vs Fourier experiments |
| `kronecker-learn/tests/test_fourier_codec.py` | Unit tests |

---

## Run

```powershell
uv sync --group dev
uv run pytest tests/test_fourier_codec.py -v
uv run python playground_v2_fourier.py
```

Building full vocab Fourier table (`d_p=16`) takes ~30–60s on first run.

---

## Expected findings (from `playground_v2_fourier.py`, `d_p=16`)

| Phenomenon | Kronecker | Fourier (measured) |
|------------|-----------|-------------------|
| Typos (`separate`/`seperate`) | 0.875 | **0.875** (match) |
| Case (`swift`/`SWIFT`) | −0.001 | **0.000** |
| Prefix (`run`/`runs`) | 0.866 | **0.866** |
| False semantics (`love`/`affection`) | 0.166 | **0.167** |
| NN of `run` | `runs`, `ru`, `runner` | **Identical top-5** |
| Mean loose morph@5 | 0.940 | **0.940** |
| Same string vs Kronecker vector | · | cosine ≈ **0.002** (different κ, same geometry) |

**Insight:** Fourier waves keyed by the same `lin_idx = byte × d_p + pos` preserve **byte-locality geometry** almost exactly after z-norm, while producing **dense** vectors (all D dims active). The encodings are different subspaces; probe rankings coincide.

**Tradeoff vs Kronecker spikes:** Dense κ uses full dimension every forward pass; may differ under training or with collision analysis; worth a Phase 5 ablation if time permits.

---

## Limitations & next steps

- Still fixed `d_p` truncation (Research #3 on dynamic positions is separate).
- No training comparison yet (plug `fourier_codec` into `KroneckerEmbedding` with a flag, or precompute Fourier table).
- Harmonics / learned frequencies could extend the basis (future work).
- Invertibility (Research #5) not addressed; dense κ may have more collisions than sparse spikes.

---

## 2-day work plan

| Day | Task |
|-----|------|
| **1** | Implement `fourier_codec.py`, tests, pairwise playground |
| **2** | Full NN probes, document results in this file, optional short training ablation |
