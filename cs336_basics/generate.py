"""Generate text from a trained checkpoint.

Usage:
    python -m cs336_basics.generate --run-dir runs/base --tokenizer data/tinystories/tokenizer.pkl \
        --prompt "Once upon a time" --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from cs336_basics.decoding import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    with open(args.tokenizer, "rb") as f:
        saved = pickle.load(f)
    tokenizer = Tokenizer(saved["vocab"], saved["merges"], saved["special_tokens"])

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        use_rope=not config.get("no_rope", False),
        use_rmsnorm=not config.get("no_rmsnorm", False),
        post_norm=config.get("post_norm", False),
        ffn_type=config.get("ffn_type", "swiglu"),
        device=args.device,
    )
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(args.prompt + text)


if __name__ == "__main__":
    main()
