# V3 Compact Factored Codec

Extension of V2: use **sin/cos for position** and **sin/cos for byte identity**, coupled in low D.

## Formula

```
byte_wave(b)[j]     = (cos(2π j b / 256) + sin(2π j b / 256)) / √2
position_wave(p)[j] = (cos(2π j p / d_p) + sin(2π j p / d_p)) / √2

bind: φ(b,p) = byte_wave(b) ⊙ position_wave(p)   # recommended
add:  φ(b,p) = byte_wave(b) + position_wave(p)

κ(token) = z_normalize( (1/√L) Σ_p φ(b_p, p) )
```

## Why bind beats add

`add` mixes byte and position linearly; collisions across `(b1,p1)` vs `(b2,p2)` are more likely.  
`bind` (Hadamard) couples them multiplicatively, closer in spirit to Kronecker's `(byte, pos)` indexing.

Measured at `d_p=16`:

| Mode | D | morph@5 |
|------|---|---------|
| Kronecker | 4096 | 0.940 |
| bind | 128 | **0.960** |
| bind | 256 | **0.950** |
| add | 256 | 0.880 |

## Run

```powershell
uv sync --group dev
uv run pytest tests/test_compact_codec.py -v
uv run python playground_v2_fourier.py   # Experiment 5 = dimension sweep
```

## Limitations

- morph@5 is a proxy; LM training not yet run
- Very low D (64) may degrade typo/prefix pairs; sweep in playground
- `add` mode included for comparison but not recommended
