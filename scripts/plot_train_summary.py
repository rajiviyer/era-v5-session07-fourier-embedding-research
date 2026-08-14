"""Build combined Layer 2 ablation figure for README."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("results")
OUT = Path("docs/figures/train_ablation_summary.png")
EMBEDDINGS = ("bpe", "kronecker", "fourier", "compact")
COLORS = {
    "bpe": "#1f77b4",
    "kronecker": "#ff7f0e",
    "fourier": "#2ca02c",
    "compact": "#d62728",
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summaries = json.loads((RESULTS / "train_summary.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for emb in EMBEDDINGS:
        rows = list(csv.DictReader((RESULTS / f"train_{emb}.csv").open(encoding="utf-8")))
        val_rows = [r for r in rows if r.get("val_loss")]
        steps = [int(r["step"]) for r in val_rows]
        vals = [float(r["val_loss"]) for r in val_rows]
        ax.plot(steps, vals, "o-", label=emb, color=COLORS[emb], linewidth=2, markersize=5)
    ax.set_xlabel("step")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title("Layer 2: validation loss (500 steps, ~80k-token corpus)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    names = [s["embedding"] for s in summaries]
    best = [s["best_val_loss"] for s in summaries]
    bars = ax2.bar(names, best, color=[COLORS[n] for n in names])
    ax2.set_ylabel("best val loss")
    ax2.set_title("Best val loss (lower is better)")
    for bar, s in zip(bars, summaries):
        label = f"{int(s['embed_params']):,} params"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    plt.close()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
