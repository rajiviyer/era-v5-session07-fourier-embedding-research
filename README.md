# Fourier & Compact Byte-Local Codecs — Research Point #4

[![CI](https://github.com/rajiviyer/era-v5-session07-fourier-embedding-research/actions/workflows/ci.yml/badge.svg)](https://github.com/rajiviyer/era-v5-session07-fourier-embedding-research/actions/workflows/ci.yml)

**Assignment problem:** What is a *real* Fourier alternative to Kronecker? Can each byte be a Fourier wave, summed to form a word?

**Answer:** Yes — in two stages:

1. **V2 Fourier codec** — replace Kronecker spikes with full-dimensional sinusoidal waves (same `D = 256 × d_p`, matches Kronecker geometry).
2. **V3 Compact codec** — factor byte and position into separate sin/cos waves, coupled via Hadamard product, reaching **comparable or better** byte-local geometry at **`D ≪ 8192`** (e.g. **D=128 at 3% of Kronecker dimensions**).

---

## Problem & motivation

Kronecker embeddings map each token to a fixed vector κ from UTF-8 bytes and within-token positions:

```
κ_K(b) = (1/√L) Σ_p  spike at index (byte_p × d_p + position_p)
```

This gives strong byte-local geometry without training the codec — but costs `D = 256 × d_p` dimensions (4096 at `d_p=16`, 8192 at `d_p=32`).

Two natural questions:

1. *Can each byte be a wave, summed to make a word?* → **V2 Fourier superposition**
2. *Can sin/cos encode position separately and keep geometry at low D?* → **V3 Compact factored codec**

---

## Solution 1: Fourier superposition (V2)

For each byte at position `p`, linear index `i_p = byte_p × d_p + p`:

```
κ_F(j) += (1/√L) · (cos(2π j i_p / D) + sin(2π j i_p / D)) / √2
```

| | Kronecker | Fourier (V2) |
|---|-----------|--------------|
| Per (byte, position) | Spike at one index | cos + sin wave across all D dims |
| Output dim | D = 256 × d_p | Same D |
| Byte-local probes | Strong | **Matches Kronecker** |

---

## Solution 2: Compact factored codec (V3)

Factor each `(byte, position)` into separate waves, then **bind** them (element-wise product — a low-rank analogue of Kronecker coupling):

```
byte_wave(b)[j]   = (cos(2π j b / 256) + sin(2π j b / 256)) / √2
position_wave(p)[j] = (cos(2π j p / d_p) + sin(2π j p / d_p)) / √2

φ(b, p) = byte_wave(b) ⊙ position_wave(p)        # Hadamard bind
κ(b)    = (1/√L) Σ_p φ(b_p, p)  →  z-normalize
```

| | Kronecker | Compact bind (V3) |
|---|-----------|-------------------|
| Output dim | 4096 (`d_p=16`) | **128–512** |
| Memory vs baseline | 100% | **3–12%** |
| Mean loose morph@5 | 0.940 | **0.950–0.960** |

**Key insight:** Position *does* carry in sin/cos — you don't need a full `d_p`-wide Kronecker axis. Factoring byte and position into shared D-dimensional waves, then binding them, preserves — and can slightly improve — morphological neighbor structure at a fraction of the dimensions.

---

## How we prove it works

Geometry probes on the codec alone (no LM training):

1. Typo robustness — `separate` / `seperate`
2. Case sensitivity — `swift` / `SWIFT`
3. Prefix locality — `run` / `runs` / `runner`
4. No false semantics — `love` / `affection`
5. NN probe — loose morph@5 on probe families
6. Dimension sweep — quality vs D (Experiment 5)

```powershell
uv sync --group dev
uv run pytest tests/ -v
uv run python playground_v2_fourier.py
```

### Results (`d_p=16`, Kronecker D=4096)

**V2 Fourier vs Kronecker** — nearly identical pairwise cosines; morph@5 = 0.940 both.

**V3 Compact bind — dimension sweep:**

| Codec | D | % of Kronecker D | mean morph@5 |
|-------|---|------------------|--------------|
| Kronecker | 4096 | 100% | 0.940 |
| Compact bind | 128 | **3.1%** | **0.960** |
| Compact bind | 256 | 6.2% | **0.950** |
| Compact bind | 512 | 12.5% | **0.950** |
| Compact add | 256 | 6.2% | 0.880 |

---

## Project structure

| File | Purpose |
|------|---------|
| `codec.py` | Kronecker spike codec (baseline) |
| `fourier_codec.py` | V2 — full-D Fourier waves |
| `compact_codec.py` | V3 — factored byte + position at low D |
| `playground_v2_fourier.py` | All experiments incl. dimension sweep |
| `tests/` | Unit tests for V2 and V3 |
| `docs/v2-fourier-alternative.md` | Detailed V2 write-up |

---

## Research horizon (full paper)

- LM training ablation at equal parameter budget
- Collision bounds for factored encodings (theory)
- Learnable frequencies on top of fixed superposition structure
- Dynamic position handling beyond fixed `d_p`

---

## References

- **Kronecker Embeddings (V1 baseline):** [kronecker-embedding-research](https://github.com/rajiviyer/kronecker-embedding-research)
- **Assignment:** `docs/ASSIGNMENT.md`
