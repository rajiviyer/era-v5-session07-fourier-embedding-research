"""
100-step smoke test: tiny GPT + codec/BPE embeddings.

    uv run python train_smoke.py
    uv run python train_smoke.py --embedding compact --steps 100
"""

from __future__ import annotations

import argparse

import torch

from embedding import EmbeddingKind
from tiny_gpt import TinyGPT, random_batch


def run_smoke(
    embedding: EmbeddingKind = "kronecker",
    steps: int = 100,
    batch_size: int = 8,
    seq_len: int = 32,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    d_p: int = 16,
    compact_dim: int = 128,
    vocab_limit: int = 1024,
    lr: float = 3e-4,
    device: str | None = None,
) -> dict[str, float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"Smoke test: embedding={embedding!r}, steps={steps}, device={device}")
    model = TinyGPT.from_embedding_kind(
        kind=embedding,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=seq_len + 1,
        d_p=d_p,
        compact_dim=compact_dim,
        vocab_limit=vocab_limit,
    ).to(dev)

    params = model.param_summary()
    print(
        f"  vocab={model.vocab_size}, d_model={d_model}, "
        f"embed_params={params['embedding_trainable']:,}, "
        f"total_params={params['total_trainable']:,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    losses: list[float] = []
    for step in range(1, steps + 1):
        input_ids, tokens = random_batch(batch_size, seq_len + 1, model.vocab_size, dev)
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]

        _, loss = model(input_ids, labels=labels)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        if step == 1 or step % 25 == 0 or step == steps:
            print(f"  step {step:3d}/{steps}  loss={loss_val:.4f}")

    first_avg = sum(losses[:5]) / 5
    last_avg = sum(losses[-5:]) / 5
    improved = last_avg < first_avg

    print(f"  first-5 avg loss={first_avg:.4f}  last-5 avg loss={last_avg:.4f}  improved={improved}")
    if not improved:
        print("  WARNING: loss did not decrease (may happen on random data; check gradients ran)")

    return {
        "first_avg_loss": first_avg,
        "last_avg_loss": last_avg,
        "improved": float(improved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny GPT embedding smoke test")
    parser.add_argument(
        "--embedding",
        choices=["bpe", "kronecker", "fourier", "compact"],
        default="kronecker",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-p", type=int, default=16)
    parser.add_argument("--compact-dim", type=int, default=128)
    parser.add_argument("--vocab-limit", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_smoke(
        embedding=args.embedding,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        d_p=args.d_p,
        compact_dim=args.compact_dim,
        vocab_limit=args.vocab_limit,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
