"""
Train tiny GPT on real text — compare BPE vs codec embeddings (assignment LM proof).

Examples:
    uv run python train.py --embedding kronecker --steps 500
    uv run python train.py --all --steps 500 --eval-every 100
    uv run python train.py --all --steps 500 --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import (
    DEFAULT_CORPUS,
    load_corpus_text,
    max_seq_len_for_corpus,
    sample_batch,
    tokenize_corpus,
    train_val_split,
)
from embedding import EmbeddingKind
from tiny_gpt import TinyGPT

RESULTS_DIR = Path("results")
ALL_EMBEDDINGS: tuple[EmbeddingKind, ...] = ("bpe", "kronecker", "fourier", "compact")


@torch.no_grad()
def evaluate(
    model: TinyGPT,
    val_ids: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    rng: np.random.Generator,
    eval_batches: int = 20,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        input_ids, labels = sample_batch(val_ids, batch_size, seq_len, device, rng)
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def train(
    embedding: EmbeddingKind = "kronecker",
    steps: int = 500,
    batch_size: int = 16,
    seq_len: int = 64,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    d_p: int = 16,
    compact_dim: int = 128,
    lr: float = 3e-4,
    val_fraction: float = 0.1,
    eval_every: int = 100,
    eval_batches: int = 20,
    max_tokens: int | None = None,
    vocab_limit: int | None = None,
    corpus_path: Path | str = DEFAULT_CORPUS,
    seed: int = 0,
    device: str | None = None,
    results_dir: Path = RESULTS_DIR,
    save_plot: bool = False,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    rng = np.random.default_rng(seed)

    text = load_corpus_text(corpus_path)
    token_ids = tokenize_corpus(text, max_tokens=max_tokens, vocab_limit=vocab_limit)
    train_ids, val_ids = train_val_split(token_ids, val_fraction=val_fraction)
    seq_len = min(seq_len, max_seq_len_for_corpus(train_ids), max_seq_len_for_corpus(val_ids))

    print(f"\n{'=' * 70}")
    print(f"Train: embedding={embedding!r}  steps={steps}  device={device}")
    print(f"  corpus tokens={len(token_ids):,}  train={len(train_ids):,}  val={len(val_ids):,}")
    print(f"  seq_len={seq_len}  batch_size={batch_size}  d_model={d_model}")

    t0 = time.perf_counter()
    model = TinyGPT.from_embedding_kind(
        kind=embedding,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=seq_len,
        d_p=d_p,
        compact_dim=compact_dim,
        vocab_limit=vocab_limit,
    ).to(dev)
    build_s = time.perf_counter() - t0
    params = model.param_summary()
    print(
        f"  vocab={model.vocab_size:,}  embed_params={params['embedding_trainable']:,}  "
        f"total_params={params['total_trainable']:,}  build={build_s:.1f}s"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    history: list[dict] = []
    best_val = float("inf")

    for step in range(1, steps + 1):
        input_ids, labels = sample_batch(train_ids, batch_size, seq_len, dev, rng)
        _, loss = model(input_ids, labels=labels)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss = float(loss.item())
        row: dict = {"step": step, "train_loss": train_loss}

        if step == 1 or step % eval_every == 0 or step == steps:
            val_loss = evaluate(
                model, val_ids, batch_size, seq_len, dev, rng, eval_batches=eval_batches
            )
            row["val_loss"] = val_loss
            best_val = min(best_val, val_loss)
            print(f"  step {step:4d}/{steps}  train={train_loss:.4f}  val={val_loss:.4f}")

        history.append(row)

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"train_{embedding}.csv"
    fieldnames = ["step", "train_loss", "val_loss"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    if best_val == float("inf"):
        best_val = evaluate(model, val_ids, batch_size, seq_len, dev, rng, eval_batches=eval_batches)

    summary = {
        "embedding": embedding,
        "steps": steps,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1].get("val_loss", best_val),
        "best_val_loss": best_val,
        "embed_params": params["embedding_trainable"],
        "total_params": params["total_trainable"],
        "vocab_size": model.vocab_size,
        "csv_path": str(csv_path),
    }

    summary_path = results_dir / f"train_{embedding}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if save_plot:
        _save_plot(history, results_dir / f"train_{embedding}.png", embedding)

    print(f"  saved {csv_path}  best_val={summary['best_val_loss']:.4f}")
    return summary


# Backward-compatible alias for tests and imports
train_ablation = train


def _save_plot(history: list[dict], path: Path, embedding: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for --plot (uv sync --group dev)") from exc

    steps = [row["step"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_steps = [row["step"] for row in history if "val_loss" in row]
    val_losses = [row["val_loss"] for row in history if "val_loss" in row]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_losses, label="train", alpha=0.7)
    if val_steps:
        plt.plot(val_steps, val_losses, "o-", label="val")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title(f"Tiny GPT — {embedding}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


def run_all(args: argparse.Namespace) -> list[dict]:
    summaries: list[dict] = []
    for kind in ALL_EMBEDDINGS:
        summaries.append(
            train(
                embedding=kind,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_p=args.d_p,
                compact_dim=args.compact_dim,
                lr=args.lr,
                val_fraction=args.val_fraction,
                eval_every=args.eval_every,
                eval_batches=args.eval_batches,
                max_tokens=args.max_tokens,
                vocab_limit=args.vocab_limit,
                corpus_path=args.corpus,
                seed=args.seed,
                device=args.device,
                results_dir=Path(args.results_dir),
                save_plot=args.plot,
            )
        )

    print(f"\n{'=' * 70}")
    print("Summary (best val loss)")
    print(f"  {'Embedding':12s}  {'Embed params':>12s}  {'Best val':>10s}")
    print("  " + "-" * 40)
    for s in summaries:
        print(f"  {s['embedding']:12s}  {s['embed_params']:12,}  {s['best_val_loss']:10.4f}")

    out = Path(args.results_dir) / "train_summary.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny GPT; compare input embeddings")
    parser.add_argument(
        "--embedding",
        choices=list(ALL_EMBEDDINGS),
        default="kronecker",
        help="Input embedding type (ignored when --all)",
    )
    parser.add_argument("--all", action="store_true", help="Run all four embedding types")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-p", type=int, default=16)
    parser.add_argument("--compact-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--vocab-limit", type=int, default=None)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.all:
        run_all(args)
    else:
        train(
            embedding=args.embedding,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_p=args.d_p,
            compact_dim=args.compact_dim,
            lr=args.lr,
            val_fraction=args.val_fraction,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            max_tokens=args.max_tokens,
            vocab_limit=args.vocab_limit,
            corpus_path=args.corpus,
            seed=args.seed,
            device=args.device,
            results_dir=Path(args.results_dir),
            save_plot=args.plot,
        )


if __name__ == "__main__":
    main()
