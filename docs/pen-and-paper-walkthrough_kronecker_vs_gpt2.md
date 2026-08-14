# Pen-and-Paper Walkthrough: BPE vs Kronecker Embeddings

Tiny dimensions you can write on one page. Same math as the repo; scaled down from GPT-2 (`V=50257`, `D=4096`, `d_model=768`).

**Related:** [learning-journal.md](learning-journal.md) (worked `"hello world"` example at full scale) · Phase 1–3 code in `kronecker-learn/`

---

## Setup: tiny toy world

| Symbol | Real GPT-2 | Paper version |
|--------|-----------|---------------|
| Vocab size `V` | 50,257 | **4** |
| `d_model` | 768 | **4** |
| `d_c` (byte rows) | 256 | **4** |
| `d_p` (max positions) | 16 | **3** |
| `D = d_c × d_p` | 4,096 | **12** |

**Toy byte alphabet** (pretend UTF-8 only uses bytes 0–3):

| Byte | Character |
|------|-----------|
| 0 | (unused) |
| 1 | `a` |
| 2 | `b` |
| 3 | `c` |

**4 tokens** (each = one BPE piece):

| Token id | String | UTF-8 bytes | Length `L` |
|----------|--------|-------------|------------|
| 0 | `ab` | `[1, 2]` | 2 |
| 1 | `ba` | `[2, 1]` | 2 |
| 2 | `abc` | `[1, 2, 3]` | 3 |
| 3 | `acb` | `[1, 3, 2]` | 3 |

**Input sentence:** two tokens → `token_ids = [0, 1]` (i.e. `"ab ba"`).

Final output shape for both paths: **`[B, T, d_model] = [1, 2, 4]`**.

---

## Part A — Regular BPE embedding

### The only matrix you learn

```
E = wte.weight   shape [V, d_model] = [4, 4]
```

Example (random init — **no structure** between rows):

```
E =  id 0 (ab)  [ -0.81  -0.11  -0.45   0.19 ]
     id 1 (ba)  [ -0.26  -0.59  -0.01   0.21 ]
     id 2 (abc) [  0.03   0.15  -0.32  -0.18 ]
     id 3 (acb) [ -0.34  -0.18  -0.41  -0.86 ]
```

### Forward pass (one lookup per token)

```
token_ids = [0, 1]

out[0] = E[0] = [-0.81, -0.11, -0.45,  0.19]   ← row for "ab"
out[1] = E[1] = [-0.26, -0.59, -0.01,  0.21]   ← row for "ba"
```

Stack → **`out` shape `[1, 2, 4]`**.

### What to notice

- `"ab"` and `"ba"` use **different rows** — no byte info is used.
- Those rows are **independent** until training connects them.
- **Trainable params:** `V × d_model = 4 × 4 = 16`.

---

## Part B — Kronecker embedding

Pipeline:

```
token_id → bytes → raw κ (sparse) → z-norm → κ (dense) → proj → [d_model]
```

Only **`proj`** is learned. κ is **fixed** from bytes.

---

### Step 1 — The 4×3 grid (before flattening)

Think of a sheet with:

- **rows** = byte value (0, 1, 2, 3)
- **cols** = position in token (0, 1, 2)

Each byte at position `p` puts a spike in cell `(byte, p)`.

**Formula** (from `codec.py`):

```
lin_idx = byte × d_p + position     (d_p = 3)
spike value = 1 / √L                  (L = number of bytes)
```

---

### Step 2 — Build raw κ for token 0: `"ab"` = bytes `[1, 2]`

`L = 2` → scale = `1/√2 ≈ 0.707`

| position | byte | lin_idx = byte×3 + pos | value |
|----------|------|------------------------|-------|
| 0 | 1 | 1×3+0 = **3** | 0.707 |
| 1 | 2 | 2×3+1 = **7** | 0.707 |

**Grid view** (rows = byte, cols = position):

```
           pos0    pos1    pos2
byte 0  [   0       0       0   ]
byte 1  [ 0.707     0       0   ]   ← 'a' at position 0
byte 2  [   0     0.707     0   ]   ← 'b' at position 1
byte 3  [   0       0       0   ]
```

**Flatten** row-by-row → vector `κ_raw` length **12**:

```
index:  0    1    2    3    4    5    6    7    8    9   10   11
κ_raw: [0,   0,   0, 0.707, 0,   0,   0, 0.707, 0,   0,   0,   0]
              ↑ spike at 3              ↑ spike at 7
```

Only **2 of 12** entries are nonzero.

---

### Step 3 — Same for token 1: `"ba"` = bytes `[2, 1]`

Same letters, **swapped order** → **different spikes**:

| position | byte | lin_idx | value |
|----------|------|---------|-------|
| 0 | 2 | **6** | 0.707 |
| 1 | 1 | **4** | 0.707 |

```
           pos0    pos1    pos2
byte 0  [   0       0       0   ]
byte 1  [   0     0.707     0   ]   ← 'a' at position 1
byte 2  [ 0.707     0       0   ]   ← 'b' at position 0
byte 3  [   0       0       0   ]
```

```
κ_raw(ba): spikes at indices 4 and 6  (not 3 and 7!)
```

**Position matters.** Same bytes as `"ab"`, different vector.

---

### Step 4 — z-normalization (makes vector dense)

Formula: subtract mean, divide by std (over all 12 coords). Paper §3.3; code in `z_normalize()`.

After z-norm, **every** entry is nonzero, but spike sites stay **large** and the rest **small**:

```
κ(ab) ≈ [-0.45, -0.45, -0.45,  2.24, -0.45, -0.45, -0.45,  2.24, -0.45, -0.45, -0.45, -0.45]
              ↑ indices 3 and 7 are the big ones
```

```
RAW:     [  0    0    0   +0.7   0    0    0   +0.7   0    0    0    0  ]
              ↓ z-norm (mean-center + scale)
Z-NORM:  [-0.45 -0.45 -0.45 +2.24 -0.45 -0.45 -0.45 +2.24 -0.45 -0.45 -0.45 -0.45]
          ↑_________________________↑
          these 10 negatives also multiply W[i,0] and affect out[0]
```

---

### Step 5 — Precomputed codec table (fixed, not trained)

Store one κ row per token id:

```
CodecTable =  shape [V, D] = [4, 12]   ← FIXED at init, not learned

CodecTable[0] = κ(ab)
CodecTable[1] = κ(ba)
CodecTable[2] = κ(abc)
CodecTable[3] = κ(acb)
```

---

### Step 6 — The only learned matrix: `proj`

From `embedding.py`:

```python
nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(self.D))
```

So **`W ~ Normal(0, 1/√D)`** — random at init, learned during training.

```
W = proj.weight   shape [D, d_model] = [12, 4]   (paper layout; PyTorch stores [d_model, D])
```

Example draw (`np.random.seed(0)`, scaled for readability):

```
W =  row0  [ 0.53   0.12   0.29   0.67 ]
     row1  [ 0.56  -0.29   0.29  -0.05 ]
     row2  [-0.03   0.12   0.04   0.44 ]
     row3  [ 0.23   0.04   0.13   0.10 ]
     row4  [ 0.45  -0.06   0.09  -0.26 ]
     row5  [-0.77   0.20   0.26  -0.22 ]
     row6  [ 0.68  -0.44   0.01  -0.06 ]
     row7  [ 0.46   0.44   0.05   0.11 ]
     row8  [-0.27  -0.59  -0.10   0.05 ]
     row9  [ 0.37   0.36  -0.12  -0.09 ]
     row10 [-0.31  -0.43  -0.51   0.59 ]
     row11 [-0.15  -0.13  -0.38   0.23 ]
```

At real scale (`D=4096`): `std = 1/√4096 ≈ 0.0156` per weight.

---

### Step 7 — Matrix multiply: embedding = κ @ W

For one token, `κ` is **1×12**, `W` is **12×4**, output is **1×4**:

```
out = κ @ W

out[j] = Σᵢ κ[i] × W[i, j]     j = 0, 1, …, d_model − 1
```

**Important:** `j` runs over **`d_model` (embedding width)**, not vocab size. Vocab size only picks **which κ row** to use.

#### Raw κ — only spike indices matter

Before z-norm, only indices 3 and 7 are nonzero:

```
out[0] = κ_raw[3]·W[3,0] + κ_raw[7]·W[7,0]
       = 0.707 × 0.23 + 0.707 × 0.46
       ≈ 0.488
```

#### Z-normed κ — all 12 indices matter

The code always projects **after** z-norm. Every coordinate contributes:

| Group | Contribution to `out[0]` |
|-------|--------------------------|
| Spike sites (indices 3, 7) | **+1.543** |
| Other 10 indices | **−0.474** |
| **Full sum** | **1.069** |

Do **not** use “spike indices only” after z-norm — it gives the wrong answer (1.54 vs 1.07).

Full z-normed vectors:

```
proj(κ_ab) ≈ [ 1.07,  1.58,  0.46, -0.11 ]
proj(κ_ba) ≈ [ 2.25, -1.05,  0.25, -1.53 ]
```

### Forward pass `token_ids = [0, 1]`

```
Kronecker out = [ proj(κ_ab),
                  proj(κ_ba) ]     shape [1, 2, 4]
```

Compare BPE vs Kronecker for the same ids:

```
BPE:       [[-0.81, -0.11, -0.45,  0.19],
            [-0.26, -0.59, -0.01,  0.21]]

Kronecker: [[ 1.07,  1.58,  0.46, -0.11],
            [ 2.25, -1.05,  0.25, -1.53]]
```

Same shape **`[1, 2, 4]`** → transformer sees identical tensor layout.

---

## Part C — Side-by-side on one page

```
Input: token_ids = [0, 1]   ("ab" then "ba")

BPE                          KRONECKER
────                          ─────────
                              bytes:
                              0→[1,2]  1→[2,1]
         ┌─────────┐                  ┌──────────────┐
ids ───► │ E[4×4]  │          ids ──►│ CodecTable   │
         │ learned │                  │ [4×12] FIXED│
         └────┬────┘                  └──────┬───────┘
              │ gather rows 0,1               │ gather κ rows 0,1
              ▼                               ▼
         [2 × 4]                          [2 × 12]
              │                               │
              │                               ▼
              │                         ┌─────────┐
              │                         │ W [12×4]│
              │                         │ LEARNED │
              │                         └────┬────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
                      [1, 2, 4]  → Transformer
```

---

## Part D — Parameter count

| | BPE | Kronecker |
|---|-----|-----------|
| Big table | `E [4×4]` **learned** | `CodecTable [4×12]` **fixed** |
| Small matrix | — | `W [12×4]` **learned** |
| **Trainable** | **16** | **48** |

In this toy example Kronecker has *more* trainable params because `V` is tiny.

The crossover happens at real scale:

```
BPE trainable:        V × d_model     = 50,257 × 768  ≈ 38.6M
Kronecker trainable:  D × d_model     =  4,096 × 768  ≈  3.1M
Kronecker fixed:      V × D           = 50,257 × 4096  (codec table, not trained)
```

**Rule of thumb:** savings appear when `V × d_model >> D × d_model`.

---

## Part E — Geometry (cosine between z-normed κ)

| Pair | Same bytes? | Same positions? | Cosine |
|------|-------------|-----------------|--------|
| `ab` vs `ba` | yes | **no** (reversed) | **−0.20** |
| `abc` vs `acb` | yes | **no** (swap) | **0.11** |
| `abc` vs `a1c` | 2/3 match | 1 typo | **0.56** |
| `ab` vs `abc` | prefix match | yes | **0.78** |

- **BPE:** `"ab"` and `"ba"` are unrelated random rows until training.
- **Kronecker:** deliberately different (order encoded); `abc` / `a1c` stay moderately similar (typo tolerance).

---

## Part F — Map back to `"hello world"` (GPT-2)

Same steps, bigger numbers:

```
"hello world" → token_ids [31373, 995]

Token 31373 ('hello'):
  bytes = [68, 65, 6c, 6c, 6f]     L = 5
  scale = 1/√5 ≈ 0.447
  spikes at lin_idx = byte×16 + position  (d_p=16)
  → κ_31373 ∈ R^4096  (fixed)
  → out[0] = W · κ_31373 ∈ R^768

Token 995 (' world'):
  bytes = [20, 77, 6f, 72, 6c, 64]   ← leading space 0x20 explicit
  → different κ_995
  → out[1] = W · κ_995 ∈ R^768
```

See [learning-journal.md](learning-journal.md) § “Worked example: hello world” for the full shape table.

---

## Part G — Index cheat sheet

```
Vocab size V     →  "which κ row?"        (lookup index)
Codec dim D      →  "how long is κ?"      (sum over i = 0 … D−1)
d_model          →  "how long is out?"    (output index j)

token_id ∈ {0, …, V−1}     picks the row
i ∈ {0, …, D−1}            indexes κ / rows of W
j ∈ {0, …, d_model−1}      indexes output embedding dimension  (NOT vocab size)
```

Vocab-sized vectors appear only at **`lm_head`** (next-token logits), downstream of embeddings.

---

## Suggested paper exercise (~15 minutes)

1. Draw the **4×3 grid** for `"ab"` and `"ba"`.
2. Write the **12-index** flattened raw vectors for each.
3. Write a **4×4** random `E` and **12×4** random `W`.
4. Compute BPE output for `ids=[0,1]` (two row lookups).
5. Compute **one** Kronecker output dimension with **all 12 terms** after z-norm, or use raw κ with **only spike indices** (two terms).

---

## Reproduce numbers in code

From repo root:

```powershell
cd kronecker-learn
python playground_phase3.py
```

Or run the toy-dimension checks from Phase 1:

```powershell
python playground.py
```

Unit tests:

```powershell
pytest tests/test_codec.py tests/test_embedding.py -v
```

---

## Common mistakes (from this walkthrough)

| Mistake | Correct |
|---------|---------|
| `out[j]` runs over vocab size | `j` runs over **`d_model`** |
| After z-norm, only spike indices matter | **All D** coordinates contribute |
| `W` values are special / derived from bytes | **`W ~ N(0, 1/√D)`** at init, then learned |
| Kronecker compresses V tokens into D numbers | Each token still has its own κ ∈ R^D; savings are **one shared W** vs V learned rows |

See also [libraries-and-pitfalls.md](libraries-and-pitfalls.md).
