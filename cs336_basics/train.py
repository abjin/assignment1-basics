"""Training loop for the Transformer language model.

Usage:
    python -m cs336_basics.train --data-dir data/tinystories --out-dir runs/base
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, get_batch, gradient_clipping, load_checkpoint, save_checkpoint
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, num_batches: int = 50) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        inputs, labels = get_batch(data, batch_size, context_length, device)
        logits = model(inputs)
        losses.append(cross_entropy(logits, labels).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with train.npy / valid.npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--total-tokens", type=int, default=327_680_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    # Ablation flags
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--post-norm", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    data_dir = Path(args.data_dir)
    train_data = np.load(data_dir / "train.npy", mmap_mode="r")
    valid_data = np.load(data_dir / "valid.npy", mmap_mode="r")

    total_steps = args.total_tokens // (args.batch_size * args.context_length)
    warmup_iters = int(total_steps * args.warmup_ratio)
    min_lr = args.lr * args.min_lr_ratio

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rope=not args.no_rope,
        use_rmsnorm=not args.no_rmsnorm,
        post_norm=args.post_norm,
        ffn_type=args.ffn_type,
        device=args.device,
    )
    if args.compile:
        model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at step {start_step}")

    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "total_steps": total_steps, "num_params": num_params}, f, indent=2)
    print(f"params={num_params / 1e6:.1f}M steps={total_steps} device={args.device}")

    model.train()
    start_time = time.time()
    best_val = math.inf

    def log(record: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    for step in range(start_step, total_steps):
        lr = get_lr_cosine_schedule(step, args.lr, min_lr, warmup_iters, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, labels = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(inputs)
        loss = cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            record = {
                "step": step,
                "train_loss": loss.item(),
                "lr": lr,
                "wall_time": elapsed,
            }
            log(record)
            print(f"step {step}/{total_steps} loss {record['train_loss']:.4f} lr {lr:.2e} t {elapsed:.0f}s")

        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, valid_data, args.batch_size, args.context_length, args.device)
            log({"step": step, "val_loss": val_loss, "wall_time": time.time() - start_time})
            print(f"step {step} val_loss {val_loss:.4f}")
            save_checkpoint(model, optimizer, step, out_dir / "latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, step, out_dir / "best.pt")

    val_loss = evaluate(model, valid_data, args.batch_size, args.context_length, args.device, num_batches=200)
    log({"step": total_steps, "val_loss": val_loss, "final": True, "wall_time": time.time() - start_time})
    print(f"final val_loss {val_loss:.4f} (best during training {best_val:.4f})")
    save_checkpoint(model, optimizer, total_steps, out_dir / "final.pt")


if __name__ == "__main__":
    main()
