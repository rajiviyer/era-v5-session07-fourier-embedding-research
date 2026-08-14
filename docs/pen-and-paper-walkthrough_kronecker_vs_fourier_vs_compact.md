# Pen-and-Paper Walkthrough: Kronecker vs Fourier vs Compact

Tiny dimensions you can write on one page. **Same toy world** as [pen-and-paper-walkthrough_kronecker_vs_gpt2.md](pen-and-paper-walkthrough_kronecker_vs_gpt2.md); here we compare all three byte-local codecs in this repo.

**Related:** [v2-fourier-alternative.md](v2-fourier-alternative.md) · [v3-compact-codec.md](v3-compact-codec.md) · code in `codec.py`, `fourier_codec.py`, `compact_codec.py`

---

## Setup: tiny toy world

| Symbol | Real scale (`d_p=16`) | Paper version |
|--------|----------------------|---------------|
| `d_c` (byte rows) | 256 | **4** |
| `d_p` (max positions) | 16 | **3** |
| Kronecker / Fourier `D` | 256 × 16 = **4096** | **12** |
| Compact `D` | 128–512 | **4** |

**Toy byte alphabet** (pretend UTF-8 only uses bytes 0–3):

| Byte | Character |
|------|-----------|
| 0 | (unused) |
| 1 | `a` |
| 2 | `b` |
| 3 | `c` |

**4 tokens:**

| Token | String | UTF-8 bytes | Length `L` |
|-------|--------|-------------|------------|
| — | `ab` | `[1, 2]` | 2 |
| — | `ba` | `[2, 1]` | 2 |
| — | `abc` | `[1, 2, 3]` | 3 |
| — | `acb` | `[1, 3, 2]` | 3 |

**Shared pipeline for all three codecs:**

```
bytes → raw κ → z-norm → κ (dense) → [optional W_proj] → d_model
```

Only the rule for building **raw κ** differs. z-normalization is identical (`z_normalize()` in `codec.py`).

---

## The one index all full-D codecs share

For Kronecker and Fourier, each byte at position `p` maps to a **linear index**:

```
lin_idx = byte × d_p + position     (d_p = 3)
scale per byte = 1 / √L
```

**4×3 grid** (row = byte, col = position); flatten row-by-row → vector length **12**:

```
              pos0    pos1    pos2
byte 0  [   0       1       2   ]
byte 1  [   3       4       5   ]
byte 2  [   6       7       8   ]
byte 3  [   9      10      11   ]
```

---

## Part A — Kronecker (spike codec)

**Rule:** put `1/√L` at exactly one coordinate per `(byte, position)`.

Reference: `kronecker_codec()` in `codec.py`.

### Token `"ab"` = bytes `[1, 2]`, `L = 2`, scale = `1/√2 ≈ 0.707`

| position | byte | lin_idx | value |
|----------|------|---------|-------|
| 0 | 1 | **3** | 0.707 |
| 1 | 2 | **7** | 0.707 |

**Grid:**

```
           pos0    pos1    pos2
byte 0  [   0       0       0   ]
byte 1  [ 0.707     0       0   ]   ← 'a' at position 0
byte 2  [   0     0.707     0   ]   ← 'b' at position 1
byte 3  [   0       0       0   ]
```

**Flattened raw κ_K** (12 entries; only 2 nonzero):

```
index:  0    1    2    3    4    5    6    7    8    9   10   11
κ_K:  [0,   0,   0, 0.707, 0,   0,   0, 0.707, 0,   0,   0,   0]
              ↑ index 3              ↑ index 7
```

### Token `"ba"` = bytes `[2, 1]` — same letters, **different order**

| position | byte | lin_idx | value |
|----------|------|---------|-------|
| 0 | 2 | **6** | 0.707 |
| 1 | 1 | **4** | 0.707 |

Spikes at indices **4 and 6** (not 3 and 7). **Position is encoded.**

### z-normalized κ_K (`"ab"`)

After subtract-mean / divide-std over all 12 coords:

```
κ_K(ab) ≈ [-0.45, -0.45, -0.45,  2.24, -0.45, -0.45, -0.45,  2.24, -0.45, -0.45, -0.45, -0.45]
                              ↑ big at 3                          ↑ big at 7
```

**Properties:** sparse raw vector; dense after z-norm; `D = d_c × d_p` dimensions.

---

## Part B — Fourier V2 (wave codec, same D)

**Rule:** same `lin_idx`, but replace each spike with a **sin/cos wave across all D coordinates**:

```
wave_i(j) = (cos(2π j i / D) + sin(2π j i / D)) / √2

κ_F(j) += (1/√L) · wave_{lin_idx}(j)
```

Reference: `_basis_wave()` and `fourier_codec()` in `fourier_codec.py`.

**Key point:** `D = 12` — **same output size as Kronecker**, drop-in replacement for `W_proj`.

### Per-byte waves for `"ab"` (each divided by √L = √2)

**lin_idx = 3** (byte 1, pos 0):

```
j:      [0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11]
wave/√L: [0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5, -0.5]
```

**lin_idx = 7** (byte 2, pos 1):

```
j:      [0,    1,     2,     3,     4,     5,     6,     7,     8,     9,    10,    11]
wave/√L: [0.5, -0.68,  0.68, -0.5,   0.18,  0.18, -0.5,   0.68, -0.68,  0.5,  -0.18, -0.18]
```

### Summed raw κ_F for `"ab"`

Add the two rows above (superposition):

```
index:  0    1     2     3     4     5     6     7     8     9    10    11
κ_F:  [1.0, -0.18, 0.18, -1.0,  0.68,  0.68, -1.0,  0.18, -0.18, 1.0, -0.68, -0.68]
```

**Every** coordinate is nonzero — dense from the start.

### z-normalized κ_F (`"ab"`)

```
κ_F(ab) ≈ [ 1.41, -0.26,  0.26, -1.41,  0.97,  0.97, -1.41,  0.26, -0.26,  1.41, -0.97, -0.97]
```

### Kronecker vs Fourier for the same token

| | Kronecker | Fourier V2 |
|---|-----------|------------|
| `D` | 12 | **12 (same)** |
| Per `(byte, pos)` | 1 spike | wave over all 12 dims |
| Raw sparsity | 2 / 12 nonzero | 12 / 12 nonzero |
| `lin_idx` formula | byte×d_p + pos | **identical** |
| Superposition | add spikes | add waves |

Same string → **different** κ vectors (spike vs sinusoid), but **similar geometry** on byte-local probes at real scale (morph@5 ≈ 0.94 for both in `README.md`).

For this toy `"ab"` example: cosine between z-normed κ_K and κ_F is about **−0.26** — correlated structure, not identical rows.

---

## Part C — Compact V3 (factored waves, low D)

**Rule:** factor byte and position into **separate** waves on a **smaller** `D`, then **bind** (Hadamard product) or **add**:

```
byte_wave(b)[j]   = (cos(2π j b / d_c) + sin(2π j b / d_c)) / √2
position_wave(p)[j] = (cos(2π j p / d_p) + sin(2π j p / d_p)) / √2

φ(b, p) = byte_wave(b) ⊙ position_wave(p)     ← bind (default)
        = byte_wave(b) + position_wave(p)     ← add (weaker)

κ_C(j) += (1/√L) · φ(b_p, p)(j)
```

Reference: `compact_codec()` in `compact_codec.py`.

Here **`D = 4`** (not 12). Output is **not** the same size as Kronecker — the savings come from sharing dimensions via factoring.

### Factor vectors (D = 4)

**byte_wave(1)** — byte `a`:

```
j:            [0,     1,     2,     3]
byte_wave(1)= [0.707, 0.707,-0.707,-0.707]
```

**position_wave(p)** — scale `d_p = 3`:

```
pos 0: [0.707, 0.707, 0.707, 0.707]
pos 1: [0.707, 0.259,-0.966, 0.707]
pos 2: [0.707,-0.966, 0.259, 0.707]
```

### Bind one `(byte, position)` pair

**φ(1, 0)** = byte_wave(1) ⊙ pos_wave(0):

```
[0.5,  0.5, -0.5, -0.5]
```

**φ(2, 1)** = byte_wave(2) ⊙ pos_wave(1) for `'b'` at position 1:

```
byte_wave(2) = [0.707, -0.707, 0.707, -0.707]
φ(2, 1)      = [0.5,  -0.183, -0.683, -0.5]
```

Position **modulates** the byte wave per dimension — this mimics Kronecker’s `(byte, pos)` coupling without a full 4×3 grid.

### Raw κ_C for `"ab"` (bind, D = 4)

Sum φ(1,0)/√2 + φ(2,1)/√2:

```
j:     [0,     1,     2,     3]
κ_C:  [0.707, 0.224,-0.837,-0.707]
```

### z-normalized κ_C (`"ab"`)

```
κ_C(ab) ≈ [ 1.34,  0.59, -1.06, -0.86]
```

Only **4** numbers to write — fits easily on one line.

### bind vs add (same `"ab"`)

| mode | raw κ_C (D=4) | Keeps byte vs position distinct? |
|------|---------------|----------------------------------|
| **bind** | `[0.71, 0.22, -0.84, -0.71]` | **yes** — `"ab"` vs `"ba"` differ |
| **add** | `[2.0, 0.68, -0.18, 0]` | **no** at this D — `"ab"` vs `"ba"` collapse (cos ≈ 1.0) |

At real scale (`D=128`, bind), add mode underperforms (morph@5 ≈ 0.88 vs 0.96 for bind in `README.md`). **Bind is the Kronecker-like choice.**

---

## Part D — Side-by-side on one page

```
Token "ab" = bytes [1, 2]

KRONECKER (D=12)              FOURIER V2 (D=12)           COMPACT bind (D=4)
────────────────              ─────────────────           ──────────────────
(byte,pos) → lin_idx          (byte,pos) → SAME lin_idx   (byte,pos) → TWO waves
         → spike 1/√L                 → sin/cos wave               → bind, then sum

[0,0,0,●,0,0,0,●,0,0,0,0]     [1,-0.2,0.2,-1,0.7,...]      [0.7, 0.2, -0.8, -0.7]
 2 nonzero                     12 nonzero                    4 nonzero
```

**ASCII pipeline:**

```
                    ┌─────────────────────────────────────────┐
bytes [1,2] ───────►│  Kronecker: spike at lin_idx 3, 7       │──► κ ∈ R^12 ──► z-norm
                    ├─────────────────────────────────────────┤
                    │  Fourier:   wave at lin_idx 3, 7        │──► κ ∈ R^12 ──► z-norm
                    ├─────────────────────────────────────────┤
                    │  Compact:   bind(byte, pos) per byte    │──► κ ∈ R^4  ──► z-norm
                    └─────────────────────────────────────────┘
```

---

## Part E — Geometry (cosine between z-normed κ)

Same pairs as the BPE/Kronecker walkthrough. Values computed with repo formulas (`d_c=4`, `d_p=3`; Compact at `D=4`, bind).

| Pair | Same bytes? | Same order? | Kronecker | Fourier | Compact bind |
|------|-------------|-------------|-----------|---------|--------------|
| `ab` vs `ba` | yes | **no** | **−0.20** | **−0.00** | **0.21** |
| `ab` vs `abc` | prefix | yes | **0.77** | **0.82** | **0.97** |
| `abc` vs `acb` | yes | **no** (swap) | **0.11** | **0.33** | **0.77** |
| `ab` vs `acb` | partial | no | 0.26 | 0.41 | 0.88 |

### How to read this table

- **Order matters:** `ab` vs `ba` should stay **dissimilar** (Kronecker −0.20). Fourier matches that (≈ 0). Compact bind stays modestly separated (0.21); **add mode fails** here (cos = 1.0 — do not use add at low D).
- **Prefix locality:** `ab` vs `abc` should stay **similar** (shared prefix bytes). All three pass; Compact bind is very high (0.97) even at D=4.
- **Permutation:** `abc` vs `acb` — same multiset of bytes, different order. Kronecker ≈ 0.11 (low, as desired). Compact bind is higher (0.77) — small D has more collision; at real `D=128` bind still beats Kronecker on morph@5.

**Toy D=4 is for algebra, not for claiming final benchmark numbers.** The qualitative pattern (bind ≈ Kronecker coupling, Fourier ≈ Kronecker at same D) is what matters on paper.

---

## Part F — Token `"abc"` (three bytes) in all three codecs

Bytes `[1, 2, 3]`, `L = 3`, scale = `1/√3 ≈ 0.577`.

| position | byte | lin_idx |
|----------|------|---------|
| 0 | 1 | **3** |
| 1 | 2 | **7** |
| 2 | 3 | **11** |

### Kronecker raw spikes

Indices **3, 7, 11** each get 0.577.

### Fourier raw

Superposition of three waves at lin_idx 3, 7, 11 → dense 12-vector; after z-norm, pattern repeats every 4 indices (three equal contributions):

```
κ_F(abc) ≈ [1.73, 0, 0, -1.73, 0, 0, -1.73, 0, 0, 1.73, 0, 0]   (z-normed, rounded)
```

### Compact bind raw (D = 4)

```
κ_C(abc) ≈ [0.87, 0.58, -0.79, -0.29]   (raw)
κ_C(abc) ≈ [1.17, 0.73, -1.33, -0.57]   (z-normed)
```

---

## Part G — Map back to real scale

| | Kronecker | Fourier V2 | Compact bind |
|---|-----------|------------|--------------|
| `D` | 256 × d_p = **4096** (d_p=16) | **4096** (same) | **128** (typical) |
| Per `(byte, pos)` | spike at `byte×d_p+pos` | sin/cos wave keyed by same index | `byte_wave ⊙ pos_wave` |
| `% of Kronecker D` | 100% | 100% | **~3%** |
| morph@5 (d_p=16) | 0.940 | 0.940 | **0.960** |

Example token `"run"` at real scale:

```
bytes = [72, 75, 6e]   L = 3
lin_idx_p = byte × 16 + position     (Kronecker & Fourier)
→ κ ∈ R^4096, then z-norm, then W_proj ∈ R^{4096 × d_model}

Compact (D=128):
→ 3 terms of bind(byte_wave, pos_wave), each ∈ R^128
→ κ ∈ R^128, then z-norm, then W_proj ∈ R^{128 × d_model}
```

---

## Part H — Index cheat sheet

```
Kronecker / Fourier:
  lin_idx = byte × d_p + position
  D       = d_c × d_p

Fourier only:
  wave_i(j) = (cos(2πji/D) + sin(2πji/D)) / √2

Compact:
  D independent (typically 128, not d_c × d_p)
  byte_wave(b), position_wave(p) each length D
  bind: φ = byte_wave ⊙ position_wave
  add:  φ = byte_wave + position_wave   (weaker coupling)

All:
  κ = (1/√L) Σ_p φ(b_p, p)  →  z-normalize
```

---

## Suggested paper exercise (~20 minutes)

1. Draw the **4×3 grid** and mark lin_idx for `"ab"` (3, 7) and `"ba"` (4, 6).
2. Write Kronecker **raw** 12-vectors for `"ab"` and `"ba"`.
3. For Fourier, compute **one** wave at lin_idx = 3 by hand for j = 0, 1, 2, 3 only:
   - phase = 2π j × 3 / 12 = π j / 4
   - j=0 → (1+0)/√2 = 0.707; j=1 → (0+1)/√2 = 0.707; j=2 → (−1+0)/√2 = −0.707; …
4. For Compact D=4, compute **bind(1, 0)** = byte_wave(1) ⊙ pos_wave(0) element-wise → `[0.5, 0.5, −0.5, −0.5]`.
5. Fill the geometry table for **one** pair you care about (e.g. `ab` vs `abc`): predict “high or low?” before checking Part E.

---

## Reproduce numbers in code

From repo root:

```powershell
uv sync --group dev
uv run python playground_v2_fourier.py
```

Unit tests:

```powershell
uv run pytest tests/test_fourier_codec.py tests/test_compact_codec.py -v
```

Quick one-off (toy dimensions):

```python
from codec import codec_from_string
from fourier_codec import fourier_codec_from_string
from compact_codec import compact_codec_from_string

d_c, d_p = 4, 3
# Note: codec_from_string uses default d_c=256; for toy bytes 0-3 pass d_c=4 explicitly
# via kronecker_codec(list("ab".encode()), d_c=4, d_p=3) after z_normalize
```

---

## Common mistakes

| Mistake | Correct |
|---------|---------|
| Fourier changes `D` | V2 Fourier keeps **D = d_c × d_p** — only the basis changes |
| Compact uses the same `lin_idx` grid | Compact **factors** byte and position; no single spike index |
| `add` and `bind` are interchangeable | **bind** (⊙) couples byte×position; add loses order at low D |
| z-norm optional | All three codecs z-norm before geometry probes and projection |
| Toy D=4 benchmarks match paper | Use toy for **algebra**; trust `playground_v2_fourier.py` for real morph@5 |

See also [pen-and-paper-walkthrough_kronecker_vs_gpt2.md](pen-and-paper-walkthrough_kronecker_vs_gpt2.md) for BPE vs Kronecker and the `W_proj` multiply step.
